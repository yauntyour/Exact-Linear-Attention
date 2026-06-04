"""
Accuracy comparison: YOLO-LAT (Hadamard Linear Attention) vs Ultralytics YOLO.
Metrics on validation set: mAP@0.5, Precision, Recall, mIoU
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from latdet.model_onnx import YOLOLAT_ONNX
from latdet.model import nms


# ── IoU ───────────────────────────────────────────────────────────────

def compute_iou_matrix(pred, gt):
    p_x1 = pred[:, 0:1] - pred[:, 2:3] / 2
    p_y1 = pred[:, 1:2] - pred[:, 3:4] / 2
    p_x2 = pred[:, 0:1] + pred[:, 2:3] / 2
    p_y2 = pred[:, 1:2] + pred[:, 3:4] / 2
    g_x1 = gt[:, 0:1] - gt[:, 2:3] / 2
    g_y1 = gt[:, 1:2] - gt[:, 3:4] / 2
    g_x2 = gt[:, 0:1] + gt[:, 2:3] / 2
    g_y2 = gt[:, 1:2] + gt[:, 3:4] / 2
    ix1 = torch.max(p_x1, g_x1.T); iy1 = torch.max(p_y1, g_y1.T)
    ix2 = torch.min(p_x2, g_x2.T); iy2 = torch.min(p_y2, g_y2.T)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    p_area = pred[:, 2] * pred[:, 3]; g_area = gt[:, 2] * gt[:, 3]
    return inter / (p_area.unsqueeze(1) + g_area.unsqueeze(0) - inter).clamp(min=1e-8)


def compute_ap(all_preds, all_gts, iou_thresh=0.5):
    """Compute AP at a given IoU threshold."""
    tp_sum = fp_sum = n_gt = 0
    for preds, gts in zip(all_preds, all_gts):
        n_gt += len(gts)
        if not preds or not gts:
            continue
        preds = sorted(preds, key=lambda x: x[4], reverse=True)
        pb = torch.tensor([p[:4] for p in preds], dtype=torch.float32)
        gb = torch.tensor(gts, dtype=torch.float32)
        iou_mat = compute_iou_matrix(pb, gb)
        matched = set()
        for m in range(len(preds)):
            best_iou, best_idx = iou_mat[m].max(0)
            if best_iou.item() >= iou_thresh and best_idx.item() not in matched:
                tp_sum += 1; matched.add(best_idx.item())
            else:
                fp_sum += 1
    prec = tp_sum / max(tp_sum + fp_sum, 1)
    rec = tp_sum / max(n_gt, 1)
    return 2 * prec * rec / (prec + rec + 1e-8), prec, rec, tp_sum, fp_sum, n_gt


# ── Evaluate ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, pairs, device, img_size=320, conf=0.25):
    model.eval()
    all_preds, all_gts = [], []
    for idx, (ip, lp) in enumerate(pairs):
        gts = []
        with open(lp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, cx, cy, w, h = map(float, parts[:5])
                    gts.append([cx, cy, w, h])
        all_gts.append(gts)

        img = cv2.imread(str(ip))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(cv2.resize(img_rgb, (img_size, img_size))
                                  ).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(device)
        out = model(tensor)[0]
        mask = out[:, 4] > conf
        dets = out[mask]
        plist = []
        if dets.numel() > 0:
            keep = nms(dets[:, :4], dets[:, 4], 0.5)
            for k in keep:
                b = dets[k, :4].tolist()
                plist.append(b + [dets[k, 4].item()])
        all_preds.append(plist)

        if (idx + 1) % 50 == 0:
            print(f'  {idx+1}/{len(pairs)}', end='\r')
    print(f'  {len(pairs)}/{len(pairs)} done.')
    return all_preds, all_gts


# ── Main ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', default='latdet/yolo26.pt')
    parser.add_argument('--lat-weights', default='latdet/runs_onnx/best.pt')
    parser.add_argument('--img-size', type=int, default=320)
    parser.add_argument('--val-split', type=int, default=255)
    parser.add_argument('--output', default='latdet/runs/benchmark_accuracy.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load val pairs
    img_dir = Path('data/images')
    label_dir = Path('data/labels')
    pairs = []
    for lp in sorted(label_dir.glob('*.txt'))[-args.val_split:]:
        ip = img_dir / f'{lp.stem}.jpg'
        if ip.exists():
            pairs.append((ip, lp))
    print(f'Validation samples: {len(pairs)}')

    # ── YOLO-LAT ──
    print('\nYOLO-LAT (Hadamard Linear Attention):')
    lat_ckpt = torch.load(args.lat_weights, map_location=device)
    nc = lat_ckpt['head.weight'].shape[0] - 5
    model_lat = YOLOLAT_ONNX(nc=nc).to(device)
    model_lat.load_state_dict(lat_ckpt)
    t0 = time.time()
    preds_lat, gts = evaluate(model_lat, pairs, device, args.img_size)
    t_lat = time.time() - t0

    # ── YOLO ──
    print('\nUltralytics YOLO:')
    from ultralytics import YOLO
    model_yolo = YOLO(args.yolo_weights)
    preds_yolo = []
    t0 = time.time()
    for idx, (ip, lp) in enumerate(pairs):
        results = model_yolo(str(ip), imgsz=args.img_size, conf=0.25, verbose=False)
        plist = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for b, s in zip(r.boxes.xywhn.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    plist.append(b.tolist() + [float(s)])
        preds_yolo.append(plist)
        if (idx + 1) % 50 == 0:
            print(f'  {idx+1}/{len(pairs)}', end='\r')
    print(f'  {len(pairs)}/{len(pairs)} done.')
    t_yolo = time.time() - t0

    # ── Metrics ──
    print('\nComputing metrics...')
    lat_params = sum(p.numel() for p in model_lat.parameters())
    yolo_params = 2504190  # from previous inspection

    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.50, 0.55, ..., 0.95
    results = {}

    for name, preds in [('YOLO-LAT (Linear Attention)', preds_lat),
                         ('Ultralytics YOLO', preds_yolo)]:
        aps = []
        precs = []
        recs = []
        for iou in iou_thresholds:
            ap, prec, rec, tp, fp, ng = compute_ap(preds, gts, iou)
            aps.append(ap)
            precs.append(prec)
            recs.append(rec)
        results[name] = {'aps': aps, 'precs': precs, 'recs': recs,
                         'map50': aps[0], 'map5095': np.mean(aps[:10]),
                         'tp': tp, 'fp': fp, 'ng': ng, 'time': t_lat if 'LAT' in name else t_yolo}

        print(f'  {name}:')
        print(f'    mAP@0.5:     {aps[0]:.3f}')
        print(f'    mAP@0.5:0.95: {np.mean(aps[:10]):.3f}')
        print(f'    Precision@0.5: {precs[0]:.3f}')
        print(f'    Recall@0.5:    {recs[0]:.3f}')
        print(f'    Time:          {results[name]["time"]:.1f}s')

    # ── Comparison table ──
    print(f'\n{"="*60}')
    print(f'{"Model":<35} {"Params":>8} {"mAP@0.5":>8} {"mAP@0.5:0.95":>12} {"Time":>7}')
    print(f'{"-"*60}')
    print(f'{"YOLO-LAT (Linear Attention)":<35} {lat_params:>8,} '
          f'{results["YOLO-LAT (Linear Attention)"]["map50"]:>8.3f} '
          f'{results["YOLO-LAT (Linear Attention)"]["map5095"]:>12.3f} '
          f'{results["YOLO-LAT (Linear Attention)"]["time"]:>6.1f}s')
    print(f'{"Ultralytics YOLO":<35} {yolo_params:>8,} '
          f'{results["Ultralytics YOLO"]["map50"]:>8.3f} '
          f'{results["Ultralytics YOLO"]["map5095"]:>12.3f} '
          f'{results["Ultralytics YOLO"]["time"]:>6.1f}s')
    print(f'{"-"*60}')

    # Speed comparison
    speedup_cpu_fps = 389 / 176  # from benchmark
    speedup_gpu_fps = 755 / 176
    params_ratio = yolo_params / lat_params
    print(f'\nSpeed-accuracy summary:')
    print(f'  YOLO-LAT is {params_ratio:.1f}x smaller ({lat_params:,} vs {yolo_params:,} params)')
    print(f'  YOLO-LAT is {speedup_cpu_fps:.1f}x faster on CPU ({389:.0f} FPS vs {176:.0f} FPS)')
    print(f'  YOLO-LAT is {speedup_gpu_fps:.1f}x faster on GPU ({755:.0f} FPS vs {176:.0f} FPS)')
    print(f'  YOLO-LAT mAP@0.5 = {results["YOLO-LAT (Linear Attention)"]["map50"]:.3f} (YOLO = {results["Ultralytics YOLO"]["map50"]:.3f})')

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    colors = {'YOLO-LAT (Linear Attention)': '#2E7D32',
              'Ultralytics YOLO': '#F44336'}
    markers = {'YOLO-LAT (Linear Attention)': 'o',
               'Ultralytics YOLO': 's'}

    # Chart 1: Precision-Recall curve (AP per IoU threshold)
    ax = axes[0]
    for name, r in results.items():
        ax.plot(iou_thresholds, r['aps'], marker=markers[name],
                color=colors[name], label=f'{name}', linewidth=2, markersize=6)
        for i, (iou, ap) in enumerate(zip(iou_thresholds, r['aps'])):
            if i % 2 == 0:
                ax.annotate(f'{ap:.2f}', (iou, ap), textcoords='offset points',
                           xytext=(0, -12), fontsize=6, ha='center', color=colors[name])
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('IoU Threshold', fontsize=11)
    ax.set_ylabel('Average Precision', fontsize=11)
    ax.set_title('AP vs IoU Threshold', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(iou_thresholds[::2])
    ax.set_ylim(0, 1.05)

    # Chart 2: Bar chart — mAP@0.5 and mAP@0.5:0.95
    ax = axes[1]
    x = np.arange(2)
    width = 0.3
    for i, (name, r) in enumerate(results.items()):
        offset = (i - 0.5) * width
        ax.bar(x + offset, [r['map50'], r['map5095']], width,
               color=colors[name], label=name, alpha=0.85)
        for j, v in enumerate([r['map50'], r['map5095']]):
            ax.annotate(f'{v:.3f}', (x[j] + offset, v), textcoords='offset points',
                       xytext=(0, 5), fontsize=8, ha='center')
    ax.set_xticks(x)
    ax.set_xticklabels(['mAP@0.5', 'mAP@0.5:0.95'], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title('mAP Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Chart 3: Bar chart — Parameters and Speed
    ax = axes[2]
    metrics_names = ['Params (K)', 'CPU FPS', 'GPU FPS']
    lat_values = [lat_params / 1000, 389, 755]
    yolo_values = [yolo_params / 1000, 176, 176]
    x = np.arange(len(metrics_names))
    width = 0.3
    for i, (name, vals) in enumerate([('YOLO-LAT (Linear Attention)', lat_values),
                                       ('Ultralytics YOLO', yolo_values)]):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, color=colors[name], label=name, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.annotate(f'{v:.0f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                       textcoords='offset points', xytext=(0, 5), fontsize=8, ha='center')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_title('Model Size & Speed', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'\nChart saved: {args.output}')


if __name__ == '__main__':
    main()
