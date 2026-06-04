"""
YOLO-LAT-ONNX: 训练 + 导出 + 基准测试 (CPU优化)

用法:
    python latdet/export_onnx.py train              # 训练模型
    python latdet/export_onnx.py export --weights latdet/runs_onnx/best.pt  # 导出ONNX
    python latdet/export_onnx.py benchmark           # 基准测试
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from latdet.model_onnx import YOLOLAT_ONNX
from latdet.model import model_size


# ── 训练 ───────────────────────────────────────────────────────────────

def train_model(args):
    """用GPU预加载方式训练 ONNX 模型 (同train.py逻辑)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    from latdet.dataset import GPUPreloadedDataset

    full = GPUPreloadedDataset(
        Path(args.data_dir) / 'images',
        Path(args.data_dir) / 'labels',
        img_size=args.img_size, device=device,
        normalize=args.normalize,
    )
    N = full.n
    n_val = int(N * args.val_split)
    n_train = N - n_val

    class _DS:
        def __init__(self, images, labels, n):
            self.images = images
            self.labels = labels
            self.n = n
        def __len__(self):
            return self.n

    train_ds = _DS(full.images[:n_train], full.labels[:n_train], n_train)
    val_ds   = _DS(full.images[n_train:], full.labels[n_train:], n_val)
    print(f'Train: {n_train}, Val: {n_val}')

    # 模型
    model = YOLOLAT_ONNX(nc=args.nc, d_model=args.d_model, n_heads=args.n_heads).to(device)
    print(f'Params: {model_size(model):,}')

    # 可选加载预训练权重
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

    # 损失函数
    from latdet.train import YOLOLoss
    criterion = YOLOLoss(nc=args.nc, cls_weight=args.cls_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    best_map = 0.0
    t_start = time.time()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t_ep = time.time()

        perm = torch.randperm(train_ds.n, device=device)
        bs = args.batch_size

        for start in range(0, train_ds.n, bs):
            end = min(start + bs, train_ds.n)
            idx = perm[start:end].tolist()
            imgs = train_ds.images[idx]
            targets = [train_ds.labels[i] for i in idx]

            # GPU增强
            from latdet.dataset import augment_gpu
            imgs, targets = augment_gpu(imgs, targets, args.img_size)

            optimizer.zero_grad()
            raw = model.forward_raw(imgs)  # (B, 5, h, w) 原始logits
            loss, comp = criterion(raw, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # 验证
        metrics = evaluate_onnx(model, val_ds, device)
        elapsed = time.time() - t_ep

        status = (f'E[{epoch:3d}/{args.epochs}] '
                  f'Loss:{epoch_loss/(train_ds.n//bs+1):.3f} '
                  f'R:{metrics["recall"]:.3f} P:{metrics["precision"]:.3f} '
                  f'mAP:{metrics["mAP"]:.3f} mIoU:{metrics["mIoU"]:.3f} '
                  f'T:{elapsed:.1f}s')
        print(status)

        if metrics['mAP'] > best_map:
            best_map = metrics['mAP']
            torch.save(model.state_dict(), save_dir / 'best.pt')
            print(f'  → New best (mAP={best_map:.3f})')

    torch.save(model.state_dict(), save_dir / 'final.pt')
    print(f'\nDone! {time.time()-t_start:.1f}s  Best mAP={best_map:.3f}')
    return save_dir / 'best.pt'


@torch.no_grad()
def evaluate_onnx(model, ds, device, conf_thresh=0.25, iou_thresh=0.5):
    """评估 (适配ONNX模型输出格式)."""
    from latdet.model import compute_iou, nms
    model.eval()
    total_gt = total_det = total_tp = 0
    iou_sum = 0.0

    N = len(ds)
    bs = 32
    for start in range(0, N, bs):
        end = min(start + bs, N)
        idx = list(range(start, end))
        imgs = ds.images[idx]
        targets = [ds.labels[i] for i in idx]

        out = model(imgs)  # (B, L, 5) — [cx, cy, w, h, score]

        for b in range(len(idx)):
            gt = targets[b]
            if gt.numel() == 0:
                continue

            dets = out[b]  # (L, 5)
            mask = dets[:, 4] > conf_thresh
            dets = dets[mask]

            if dets.numel() == 0:
                total_gt += gt.shape[0]
                continue

            boxes_pred = dets[:, :4]
            scores_pred = dets[:, 4]
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
    return {'recall': recall, 'precision': precision,
            'mAP': (recall + precision) / 2, 'mIoU': iou_sum / max(total_tp, 1)}


# ── ONNX 导出 ─────────────────────────────────────────────────────────

def export_onnx(args):
    """BN融合 → ONNX导出."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.weights, map_location=device)
    d_model = ckpt['neck_proj.conv.weight'].shape[0]  # output channels = d_model
    model = YOLOLAT_ONNX(nc=args.nc, d_model=d_model).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f'Loaded weights: {args.weights}')

    # 导出ONNX (do_constant_folding自动处理BN折叠)
    model = model.cpu().eval()
    save_path = Path(args.save_dir) / 'yololat.onnx'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    x = torch.randn(1, 3, args.img_size, args.img_size)
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 1: 'num_detections'},
    }
    torch.onnx.export(
        model,
        x,
        str(save_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    print(f'ONNX exported: {save_path}  ({save_path.stat().st_size / 1024:.0f} KB)')

    # 验证ONNX
    try:
        import onnx
        onnx_model = onnx.load(str(save_path))
        onnx.checker.check_model(onnx_model)
        print('ONNX check: PASS')
    except ImportError:
        print('onnx package not installed, skipping check')
    except Exception as e:
        print(f'ONNX check: FAIL — {e}')

    # ONNX Runtime 测试
    try:
        import onnxruntime as ort
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(str(save_path), providers=providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # 多分辨率测试
        for h, w in [(224, 224), (320, 320), (416, 416), (640, 480)]:
            inp = np.random.randn(1, 3, h, w).astype(np.float32)
            out = session.run([output_name], {input_name: inp})[0]
            print(f'  ONNX Runtime: {h}x{w} → {list(out.shape)}')

        # 基准
        print('\nONNX Runtime Benchmark (CPU):')
        inp = np.random.randn(1, 3, 320, 320).astype(np.float32)
        for _ in range(50):
            session.run([output_name], {input_name: inp})
        t0 = time.perf_counter()
        n_iter = 500
        for _ in range(n_iter):
            session.run([output_name], {input_name: inp})
        t1 = time.perf_counter()
        avg_ms = (t1 - t0) / n_iter * 1000
        print(f'  {n_iter} iters: {avg_ms:.2f}ms avg  ({1000/avg_ms:.0f} FPS)')
    except ImportError:
        print('onnxruntime not installed, skipping runtime test')

    return str(save_path)


def fuse_bn_into_conv(model):
    """深度融合BN → Conv权重."""
    model = model.cpu()
    model.eval()
    model = model.to('cpu')
    for name, module in list(model.named_children()):
        if isinstance(module, (nn.BatchNorm2d,)):
            continue
        if hasattr(module, 'conv') and hasattr(module, 'bn'):
            _fuse(module.conv, module.bn)
        elif hasattr(module, 'qkv') and module.qkv is not None:
            continue  # qkv没有bn
        elif hasattr(module, 'proj') and module.proj is not None:
            continue
        elif isinstance(module, nn.Sequential):
            for sub in module:
                if hasattr(sub, 'conv') and hasattr(sub, 'bn'):
                    _fuse(sub.conv, sub.bn)
                if hasattr(sub, 'bn') and isinstance(sub.bn, nn.BatchNorm2d):
                    _fuse(sub.conv, sub.bn)
    return model


def _fuse(conv, bn):
    """融合单对 Conv+BN."""
    if bn is None or isinstance(bn, nn.Identity):
        return
    w = conv.weight
    mean = bn.running_mean
    var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (var + eps).sqrt()
    scale = gamma / std
    conv.weight.data = w * scale.view(-1, 1, 1, 1)
    if conv.bias is None:
        conv.bias = nn.Parameter(torch.zeros_like(beta))
        conv.bias.requires_grad = False
    conv.bias.data = beta - mean * scale
    # 清除BN (推理时跳过)
    bn.weight.data = torch.ones_like(gamma)
    bn.bias.data = torch.zeros_like(beta)
    bn.running_mean.data = torch.zeros_like(mean)
    bn.running_var.data = torch.ones_like(var)


# ── 基准 (PyTorch CPU) ────────────────────────────────────────────────

def benchmark(args):
    device = torch.device('cpu')
    model = YOLOLAT_ONNX(nc=args.nc).to(device)
    ckpt = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    print(f'PyTorch CPU Benchmark ({args.img_size}x{args.img_size}):')
    for bs in [1, 4, 8, 16]:
        x = torch.randn(bs, 3, args.img_size, args.img_size)
        for _ in range(30):
            model(x)
        t0 = time.perf_counter()
        for _ in range(200):
            model(x)
        t1 = time.perf_counter()
        avg_ms = (t1 - t0) / 200 * 1000
        fps = 200 / (t1 - t0)
        print(f'  batch={bs:2d}:  {avg_ms:.2f}ms  ({fps:.0f} FPS)')


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='YOLO-LAT-ONNX')
    parser.add_argument('mode', choices=['train', 'export', 'benchmark', 'all'],
                       help='train: 训练 | export: 导ONNX | benchmark: 基准 | all: 全流程')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--save-dir', default='latdet/runs_onnx')
    parser.add_argument('--weights', default='latdet/runs_onnx/best.pt')
    parser.add_argument('--img-size', type=int, default=320)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=60)
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

    if args.mode in ('train', 'all'):
        print('='*50, '\nTRAINING\n', '='*50)
        best_pt = train_model(args)
        args.weights = str(best_pt)

    if args.mode in ('export', 'all'):
        print('\n' + '='*50 + '\nEXPORT\n' + '='*50)
        export_onnx(args)

    if args.mode in ('benchmark', 'all'):
        print('\n' + '='*50 + '\nBENCHMARK\n' + '='*50)
        benchmark(args)
