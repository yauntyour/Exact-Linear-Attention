"""
Compare 3 models: YOLO-LAT (Hadamard linear attention) vs YOLO-LAT (softmax) vs Ultralytics YOLO.

Tests:
  1. Latency & FPS at various batch sizes (CPU & GPU)
  2. Resolution scaling (how attention complexity affects latency as input grows)
"""
import sys, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from latdet.model_onnx import YOLOLAT_ONNX, YOLOLAT_Softmax
from latdet.model import model_size

# ── Benchmark helpers ─────────────────────────────────────────────────

@torch.no_grad()
def benchmark_pt(model, device, img_size, batch_sizes, n_warmup=50, n_iter=300):
    """PyTorch model benchmark."""
    results = {}
    for bs in batch_sizes:
        dummy = torch.randn(bs, 3, img_size, img_size).to(device)
        for _ in range(n_warmup):
            model(dummy)
        if device.type == 'cuda':
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(n_iter):
                model(dummy)
            end_ev.record()
            torch.cuda.synchronize()
            total_ms = start_ev.elapsed_time(end_ev) / n_iter
        else:
            t0 = time.perf_counter()
            for _ in range(n_iter):
                model(dummy)
            t1 = time.perf_counter()
            total_ms = (t1 - t0) / n_iter * 1000
        fps = 1000 / max(total_ms, 1e-6)
        results[bs] = {'avg_ms': total_ms, 'fps': fps}
    return results


def benchmark_yolo(model_yolo, device, img_size, batch_sizes, n_warmup=50, n_iter=300):
    """Ultralytics YOLO benchmark."""
    results = {}
    for bs in batch_sizes:
        dummy = [np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
                 for _ in range(bs)]
        for _ in range(n_warmup):
            model_yolo(dummy, verbose=False)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model_yolo(dummy, verbose=False)
        t1 = time.perf_counter()
        total_ms = (t1 - t0) / n_iter * 1000
        fps = 1000 / max(total_ms, 1e-6)
        results[bs] = {'avg_ms': total_ms, 'fps': fps}
    return results


def benchmark_resolution(model, device, img_sizes, n_warmup=30, n_iter=200):
    """Measure latency at different resolutions to show attention scaling."""
    results = {}
    for size in img_sizes:
        dummy = torch.randn(1, 3, size, size).to(device)
        for _ in range(n_warmup):
            model(dummy)
        if device.type == 'cuda':
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(n_iter):
                model(dummy)
            end_ev.record()
            torch.cuda.synchronize()
            ms = start_ev.elapsed_time(end_ev) / n_iter
        else:
            t0 = time.perf_counter()
            for _ in range(n_iter):
                model(dummy)
            t1 = time.perf_counter()
            ms = (t1 - t0) / n_iter * 1000
        L = (size // 16) ** 2
        results[size] = {'avg_ms': ms, 'fps': 1000/ms, 'L': L}
    return results


# ── Plot style ────────────────────────────────────────────────────────

COLORS = {
    'YOLO-LAT (Linear Attention)': '#2E7D32',
    'YOLO-LAT (Softmax Attention)': '#E65100',
    'Ultralytics YOLO': '#F44336',
}
MARKERS = {
    'YOLO-LAT (Linear Attention)': 'o',
    'YOLO-LAT (Softmax Attention)': 's',
    'Ultralytics YOLO': 'D',
}


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='3-way model comparison')
    parser.add_argument('--yolo-weights', default='latdet/yolo26.pt')
    parser.add_argument('--lat-weights', default='latdet/runs_onnx/best.pt')
    parser.add_argument('--img-size', type=int, default=320)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8, 16])
    parser.add_argument('--output', default='latdet/runs/benchmark_3way.png')
    args = parser.parse_args()

    # ── Load models ──
    print('Loading models...')
    lat_ckpt = torch.load(args.lat_weights, map_location='cpu')
    nc = lat_ckpt['head.weight'].shape[0] - 5

    model_linear = YOLOLAT_ONNX(nc=nc).eval()
    model_linear.load_state_dict(lat_ckpt)
    print(f'  YOLO-LAT (Linear Attention):    {model_size(model_linear):,} params')

    model_softmax = YOLOLAT_Softmax(nc=nc).eval()
    model_softmax.load_state_dict(lat_ckpt, strict=False)
    print(f'  YOLO-LAT (Softmax Attention):   {model_size(model_softmax):,} params')

    from ultralytics import YOLO
    model_yolo = YOLO(args.yolo_weights)
    yolo_params = sum(p.numel() for p in model_yolo.model.parameters() if p.requires_grad)
    print(f'  Ultralytics YOLO:                {yolo_params:,} params')

    results = {}
    res_scale = {}

    for device_name, device in [('CPU', torch.device('cpu'))]:
        print(f'\n=== {device_name} ===')

        for label, model in [
            ('YOLO-LAT (Linear Attention)', model_linear),
            ('YOLO-LAT (Softmax Attention)', model_softmax),
        ]:
            m = model.to(device)
            # batch-size benchmark
            r = benchmark_pt(m, device, args.img_size, args.batch_sizes)
            results[f'{label} ({device_name})'] = r
            print(f'\n  {label}:')
            for bs, v in r.items():
                print(f'    batch={bs:2d}:  {v["avg_ms"]:.2f}ms  {v["fps"]:.0f} FPS')

            # resolution scaling benchmark
            sizes = [160, 224, 320, 416, 512, 640]
            rr = benchmark_resolution(m, device, sizes)
            res_scale[f'{label} ({device_name})'] = rr
            print(f'  Resolution scaling:')
            for s, v in rr.items():
                L = v['L']
                print(f'    {s}x{s}: L={L:4d}  {v["avg_ms"]:.2f}ms  {v["fps"]:.0f} FPS')

        # YOLO
        r = benchmark_yolo(model_yolo, device, args.img_size, args.batch_sizes)
        results[f'Ultralytics YOLO ({device_name})'] = r
        print(f'\n  Ultralytics YOLO:')
        for bs, v in r.items():
            print(f'    batch={bs:2d}:  {v["avg_ms"]:.2f}ms  {v["fps"]:.0f} FPS')

    # ── GPU ──
    if torch.cuda.is_available():
        device_name = 'GPU'
        device = torch.device('cuda')
        print(f'\n=== {device_name} ===')

        for label, model in [
            ('YOLO-LAT (Linear Attention)', model_linear),
            ('YOLO-LAT (Softmax Attention)', model_softmax),
        ]:
            m = model.to(device)
            r = benchmark_pt(m, device, args.img_size, args.batch_sizes)
            results[f'{label} ({device_name})'] = r
            print(f'\n  {label}:')
            for bs, v in r.items():
                print(f'    batch={bs:2d}:  {v["avg_ms"]:.2f}ms  {v["fps"]:.0f} FPS')

            sizes = [160, 224, 320, 416, 512, 640]
            rr = benchmark_resolution(m, device, sizes)
            res_scale[f'{label} ({device_name})'] = rr

        r = benchmark_yolo(model_yolo, device, args.img_size, args.batch_sizes)
        results[f'Ultralytics YOLO ({device_name})'] = r
        print(f'\n  Ultralytics YOLO:')
        for bs, v in r.items():
            print(f'    batch={bs:2d}:  {v["avg_ms"]:.2f}ms  {v["fps"]:.0f} FPS')

    # ── Plot 2x2 ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('YOLO-LAT: Linear Attention vs Softmax Attention vs Ultralytics YOLO',
                 fontsize=14, fontweight='bold', y=1.01)

    # Plot 1: Batch-size FPS
    ax = axes[0, 0]
    for label, r in results.items():
        if 'CPU' not in label:
            continue
        bs_list = sorted(r.keys())
        fps = [r[b]['fps'] for b in bs_list]
        name = label.replace(' (CPU)', '')
        ax.plot(bs_list, fps, marker=MARKERS.get(name, 'o'),
                color=COLORS.get(name, '#888'), label=name,
                linewidth=2, markersize=8)
        for b, f in zip(bs_list, fps):
            ax.annotate(f'{f:.0f}', (b, f), textcoords='offset points',
                        xytext=(0, 12), fontsize=7, ha='center', color=COLORS.get(name, '#888'))
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('FPS', fontsize=12)
    ax.set_title('CPU Throughput (higher is better)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted(results[list(results.keys())[0]].keys()))

    # Plot 2: Batch-size FPS (GPU)
    ax = axes[0, 1]
    for label, r in results.items():
        if 'GPU' not in label:
            continue
        bs_list = sorted(r.keys())
        fps = [r[b]['fps'] for b in bs_list]
        name = label.replace(' (GPU)', '')
        ax.plot(bs_list, fps, marker=MARKERS.get(name, 'o'),
                color=COLORS.get(name, '#888'), label=name,
                linewidth=2, markersize=8)
        for b, f in zip(bs_list, fps):
            ax.annotate(f'{f:.0f}', (b, f), textcoords='offset points',
                        xytext=(0, 12), fontsize=7, ha='center', color=COLORS.get(name, '#888'))
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('FPS', fontsize=12)
    ax.set_title('GPU Throughput (higher is better)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted(results[list(results.keys())[0]].keys()))

    # Plot 3: Resolution scaling (CPU)
    ax = axes[1, 0]
    for label, rr in res_scale.items():
        if 'CPU' not in label:
            continue
        sizes = sorted(rr.keys())
        lat = [rr[s]['avg_ms'] for s in sizes]
        L_vals = [rr[s]['L'] for s in sizes]
        name = label.replace(' (CPU)', '')
        ax.plot(sizes, lat, marker=MARKERS.get(name, 'o'),
                color=COLORS.get(name, '#888'), label=name,
                linewidth=2, markersize=8)
        # Annotate with L and ratio
        for s, l, Lv in zip(sizes, lat, L_vals):
            ax.annotate(f'L={Lv}', (s, l), textcoords='offset points',
                        xytext=(0, 10), fontsize=6, ha='center', color=COLORS.get(name, '#888'))
    ax.set_xlabel('Input Size', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('CPU: Resolution Scaling (lower is better)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([160, 224, 320, 416, 512, 640])
    ax.set_xticklabels(['160', '224', '320', '416', '512', '640'])

    # Plot 4: Latency vs L (attention positions) to show O(L²) vs O(L)
    ax = axes[1, 1]
    for label, rr in res_scale.items():
        if 'CPU' not in label:
            continue
        sizes = sorted(rr.keys())
        L_vals = [rr[s]['L'] for s in sizes]
        lat = [rr[s]['avg_ms'] for s in sizes]
        name = label.replace(' (CPU)', '')
        ax.plot(L_vals, lat, marker=MARKERS.get(name, 'o'),
                color=COLORS.get(name, '#888'), label=name,
                linewidth=2, markersize=8)
        # Fit polynomial to show trend
        z = np.polyfit(L_vals, lat, 2)
        trend = np.poly1d(z)
        L_fit = np.linspace(min(L_vals), max(L_vals), 100)
        ax.plot(L_fit, trend(L_fit), '--', color=COLORS.get(name, '#888'), alpha=0.3, linewidth=1)
        # Annotate
        for Lv, l in zip(L_vals, lat):
            ax.annotate(f'{l:.1f}ms', (Lv, l), textcoords='offset points',
                        xytext=(5, 5), fontsize=7, color=COLORS.get(name, '#888'))
    # Reference lines for O(L) and O(L²)
    ax.text(0.3, 0.95, 'Linear attention scales as O(L·d²) → nearly linear',
            transform=ax.transAxes, fontsize=9, color='#2E7D32', va='top',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))
    ax.text(0.3, 0.85, 'Softmax attention scales as O(L²·d) → quadratic',
            transform=ax.transAxes, fontsize=9, color='#E65100', va='top',
            bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.8))
    ax.set_xlabel('Number of positions L = (H/16)²', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('CPU: Latency vs Attention Positions (lower is better)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Info footer
    fig.text(0.5, 0.01,
             f'Input: {args.img_size}x{args.img_size}  |  '
             f'Linear: {model_size(model_linear):,} params  |  '
             f'Softmax: {model_size(model_softmax):,} params  |  '
             f'YOLO: {yolo_params:,} params',
             ha='center', fontsize=9, color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'\nChart saved: {args.output}')

    # ── Summary table ──
    print(f'\n{"="*75}')
    print(f'{"Model":<35} {"Device":<6} {"Batch=1":>13} {"Batch=8":>13} {"Batch=16":>13}')
    print(f'{"-"*75}')
    for label, r in results.items():
        parts = label.rsplit(' (', 1)
        name = parts[0]
        dev = parts[1].rstrip(')')
        b1 = r.get(1, {}); b8 = r.get(8, {}); b16 = r.get(16, {})
        print(f'{name:<35} {dev:<6} '
              f'{b1.get("fps",0):>5.0f} FPS ({b1.get("avg_ms",0):.1f}ms)  '
              f'{b8.get("fps",0):>5.0f} FPS ({b8.get("avg_ms",0):.1f}ms)  '
              f'{b16.get("fps",0):>5.0f} FPS ({b16.get("avg_ms",0):.1f}ms)')
    print(f'{"="*75}')

    # Complexity comparison
    print(f'\nComplexity analysis (d_model=64, n_heads=4, head_dim=16):')
    print(f'{"Resolution":<12} {"L":<6} {"Linear O(L·D²)":<20} {"Softmax O(L²·D)":<20} {"Ratio":<8}')
    print(f'{"-"*66}')
    for s in [160, 224, 320, 416, 512, 640]:
        L = (s // 16) ** 2
        D = 64
        lin_macs = L * D * D / 1e6
        soft_macs = L * L * D / 1e6
        ratio = soft_macs / lin_macs if lin_macs > 0 else 0
        print(f'{s}x{s:<5}     {L:<6} {lin_macs:<19.2f}M {soft_macs:<19.2f}M {ratio:<7.1f}x')


if __name__ == '__main__':
    main()
