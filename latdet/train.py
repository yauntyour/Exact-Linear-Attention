"""
YOLO-LAT training with GPU-preloaded data.

All images reside on GPU as a single tensor — zero CPU-GPU transfer during training.
Augmentations (flip, HSV) are performed on GPU via tensor ops.
"""
import sys
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latdet.model import YOLOLAT, compute_iou, nms, model_size
from latdet.dataset import GPUPreloadedDataset, augment_gpu


# ── Loss ───────────────────────────────────────────────────────────────

class YOLOLoss(nn.Module):
    """
    Detection loss: Focal (obj) + CIoU (box) + BCE (class).

    All computation on GPU, no CPU round-trips.
    """
    def __init__(self, nc=1, cls_weight=1.0):
        super().__init__()
        self.nc = nc
        self.cls_weight = cls_weight

    def forward(self, preds, targets):
        """
        preds:   (B, 5+nc, H, W) raw output
        targets: list of B tensors (n_i, 5)  [cls_id, cx, cy, w, h]
        """
        B, C, H, W = preds.shape
        device = preds.device

        # Target tensors on device
        obj_target = torch.zeros(B, 1, H, W, device=device)
        box_target = torch.zeros(B, 4, H, W, device=device)
        pos_count = 0

        # Grid
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij',
        )
        xs = xs.float().view(1, 1, H, W)
        ys = ys.float().view(1, 1, H, W)

        for b in range(B):
            tgt = targets[b]
            if tgt.numel() == 0:
                continue
            for row in tgt:
                cls_id, cx, cy, w, h = row.tolist()
                gi = int(cx * W)
                gj = int(cy * H)
                if 0 <= gi < W and 0 <= gj < H:
                    obj_target[b, 0, gj, gi] = 1.0
                    box_target[b, 0, gj, gi] = cx
                    box_target[b, 1, gj, gi] = cy
                    box_target[b, 2, gj, gi] = w
                    box_target[b, 3, gj, gi] = h
                    pos_count += 1

        # ── 1. Objectness Focal Loss ──
        obj_logits = preds[:, 0:1]
        p = obj_logits.sigmoid()
        pos_mask = obj_target > 0
        neg_mask = ~pos_mask
        alpha, gamma = 0.25, 2.0
        focal_w = alpha * (1 - p) ** gamma * pos_mask.float() \
                + (1 - alpha) * p ** gamma * neg_mask.float()
        obj_loss = F.binary_cross_entropy_with_logits(
            obj_logits, obj_target, reduction='none'
        )
        obj_loss = (obj_loss * focal_w).sum() / max(pos_count, 1)

        # ── 2. CIoU Box Loss ──
        box_loss = torch.tensor(0.0, device=device)
        if pos_count > 0:
            pred_xy = preds[:, 1:3].sigmoid()
            pred_wh = preds[:, 3:5].sigmoid()
            pred_cx = (xs + pred_xy[:, 0:1]) / W
            pred_cy = (ys + pred_xy[:, 1:2]) / H
            pred_w  = (pred_wh[:, 0:1] * 2) / W
            pred_h  = (pred_wh[:, 1:2] * 2) / H
            decoded = torch.cat([pred_cx, pred_cy, pred_w, pred_h], dim=1)

            pos_expand = pos_mask.expand(-1, 4, -1, -1)
            pos_pred = decoded[pos_expand].reshape(-1, 4)
            pos_tgt  = box_target[pos_expand].reshape(-1, 4)
            if pos_pred.numel() > 0:
                ciou = compute_iou(pos_pred, pos_tgt, mode='ciou')
                box_loss = (1.0 - ciou).mean()

        # ── 3. Class Loss ──
        cls_loss = torch.tensor(0.0, device=device)
        if self.nc > 0 and pos_count > 0:
            cls_lg = preds[:, 5:]
            cls_tg = torch.zeros_like(cls_lg)
            for b in range(B):
                for row in targets[b]:
                    cls_id, cx, cy = int(row[0].item()), row[1].item(), row[2].item()
                    gi, gj = int(cx * W), int(cy * H)
                    if 0 <= gi < W and 0 <= gj < H:
                        cls_tg[b, cls_id, gj, gi] = 1.0
            cls_loss = F.binary_cross_entropy_with_logits(cls_lg, cls_tg, reduction='sum')
            cls_loss = cls_loss / max(pos_count, 1) * self.cls_weight

        total = obj_loss + box_loss + cls_loss
        return total, {'obj': obj_loss.item(), 'box': box_loss.item(),
                       'cls': cls_loss.item(), 'pos': pos_count}


# ── Evaluation ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataset, device, conf_thresh=0.25, iou_thresh=0.5):
    """Run through entire dataset evaluating recall/precision."""
    model.eval()
    total_gt = total_det = total_tp = 0
    iou_sum = 0.0

    # Process in batches
    N = len(dataset)
    batch_size = 32
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        idx = list(range(start, end))
        imgs = dataset.images[idx]
        targets = [dataset.labels[i] for i in idx]

        preds = model(imgs)
        decoded = model.module.decode(preds, conf_thresh=conf_thresh) if hasattr(model, 'module') else model.decode(preds, conf_thresh=conf_thresh)

        for b in range(len(idx)):
            gt = targets[b]
            if gt.numel() == 0:
                continue
            boxes_pred, scores_pred, _ = decoded[b]
            if boxes_pred.numel() == 0:
                total_gt += gt.shape[0]
                continue
            keep = nms(boxes_pred, scores_pred, iou_thresh=0.5)
            boxes_pred = boxes_pred[keep] if keep.numel() > 0 else boxes_pred
            if boxes_pred.numel() == 0:
                total_gt += gt.shape[0]
                continue

            gt_boxes = gt[:, 1:5].to(boxes_pred.device)
            matched = set()
            tp = iou_acc = 0
            for pb in boxes_pred:
                ious = compute_iou(pb.unsqueeze(0), gt_boxes, mode='iou')
                best_iou, best_idx = ious.max(0)
                if best_iou.item() > iou_thresh and best_idx.item() not in matched:
                    tp += 1
                    iou_acc += best_iou.item()
                    matched.add(best_idx.item())
            total_gt += gt.shape[0]
            total_det += boxes_pred.shape[0]
            total_tp += tp
            iou_sum += iou_acc

    recall = total_tp / max(total_gt, 1)
    precision = total_tp / max(total_det, 1)
    return {
        'recall': recall, 'precision': precision,
        'mAP': (recall + precision) / 2 if (recall + precision) > 0 else 0,
        'mIoU': iou_sum / max(total_tp, 1),
    }


# ── Training ───────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Preload ALL data to GPU ──
    full = GPUPreloadedDataset(args.data_dir / 'images', args.data_dir / 'labels',
                               img_size=args.img_size, device=device,
                               normalize=args.normalize)
    N = full.n
    n_val = int(N * args.val_split)
    n_train = N - n_val

    # Split: train = first n_train, val = last n_val
    train_idx = list(range(n_train))
    val_idx   = list(range(n_train, N))

    class _SplitDS:
        """Lightweight namespace for a GPU-resident split."""
        def __init__(self, images, labels, n):
            self.images = images
            self.labels = labels
            self.n = n
        def __len__(self):
            return self.n

    train_ds = _SplitDS(full.images[train_idx],
                         [full.labels[i] for i in train_idx], n_train)
    val_ds   = _SplitDS(full.images[val_idx],
                         [full.labels[i] for i in val_idx], n_val)

    print(f'Train: {n_train}, Val: {n_val}')

    # ── Model ──
    model = YOLOLAT(nc=args.nc, d_model=args.d_model, n_heads=args.n_heads).to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print(f'Params: {model_size(model):,}')

    # Optional pretrained weights
    if args.pretrained:
        state = torch.load(args.pretrained, map_location=device)
        model_state = model.state_dict()
        loaded = skipped = 0
        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                model_state[k] = v
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(model_state)
        print(f'Loaded pretrained: {loaded} layers, skipped: {skipped}')

    criterion = YOLOLoss(nc=args.nc, cls_weight=args.cls_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ── Train loop ──
    best_map = 0.0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t_ep = time.time()

        # Shuffle training indices — pure GPU operation
        perm = torch.randperm(train_ds.n, device=device)
        bs = args.batch_size

        for start in range(0, train_ds.n, bs):
            end = min(start + bs, train_ds.n)
            batch_idx = perm[start:end].tolist()

            # Slice directly from GPU tensor — zero CPU transfer for images
            imgs = train_ds.images[batch_idx]  # (B, 3, H, W) already on GPU
            targets = [train_ds.labels[i] for i in batch_idx]

            # GPU augmentations
            imgs, targets = augment_gpu(imgs, targets, args.img_size)

            optimizer.zero_grad()
            preds = model(imgs)
            loss, components = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        # Validation every epoch
        metrics = evaluate(model, val_ds, device)

        elapsed = time.time() - t_ep
        lr_now = scheduler.get_last_lr()[0]
        print(
            f'E[{epoch:3d}/{args.epochs}] '
            f'Loss:{epoch_loss/(train_ds.n//bs+1):.3f} '
            f'O:{components["obj"]:.3f} '
            f'B:{components["box"]:.3f} '
            f'C:{components["cls"]:.3f} '
            f'R:{metrics["recall"]:.3f} '
            f'P:{metrics["precision"]:.3f} '
            f'mAP:{metrics["mAP"]:.3f} '
            f'mIoU:{metrics["mIoU"]:.3f} '
            f'LR:{lr_now:.2e} '
            f'T:{elapsed:.1f}s'
        )

        if metrics['mAP'] > best_map:
            best_map = metrics['mAP']
            state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state, args.save_dir / 'best.pt')
            print(f'  → New best (mAP={best_map:.3f})')

    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state, args.save_dir / 'final.pt')
    print(f'\nDone! {time.time()-t_start:.1f}s  Best mAP={best_map:.3f}')
    return best_map


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='YOLO-LAT Training (GPU-preloaded)')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--save-dir', default='latdet/runs')
    parser.add_argument('--img-size', type=int, default=320)
    parser.add_argument('--batch-size', type=int, default=128, help='large batches since data is on GPU')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--nc', type=int, default=1)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--pretrained', type=str, default=None, help='path to pretrained weights')
    parser.add_argument('--normalize', action='store_true', help='per-image contrast stretch')
    parser.add_argument('--cls-weight', type=float, default=1.0, help='class loss weight multiplier')

    args = parser.parse_args()
    args.data_dir = Path(args.data_dir).resolve()
    if not args.data_dir.exists():
        args.data_dir = Path(__file__).resolve().parent.parent / 'data'
    args.save_dir = Path(args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    train(args)
