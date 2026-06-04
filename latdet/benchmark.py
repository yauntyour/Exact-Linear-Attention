"""
YOLO-LAT 工业级性能基准测试

测试项:
  1. 端到端延迟 (预处理 + 推理 + 解码 + NMS)
  2. 不同 batch size 的吞吐量
  3. GPU / CPU 对比
  4. 延迟分布 (P50 / P95 / P99)
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import cv2

from latdet.model import YOLOLAT, nms


def format_ns(ns):
    """ns → 带单位字符串"""
    if ns < 1_000:
        return f'{ns:.1f}ns'
    elif ns < 1_000_000:
        return f'{ns/1_000:.1f}us'
    else:
        return f'{ns/1_000_000:.1f}ms'


@torch.no_grad()
def benchmark(model, device, img_size, batch_sizes, n_warmup=100, n_iter=500):
    """
    完整pipeline基准测试: preprocess → infer → decode → nms

    返回: {batch_size: {avg_ms, fps, p50, p95, p99, ...}}
    """
    print(f'\n{"="*50}')
    print(f'Device: {device}')
    print(f'Image size: {img_size}')
    print(f'{"="*50}')

    results = {}

    for bs in batch_sizes:
        # 构造批量输入 (单张图重复, 模拟真实场景)
        dummy_img = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        latencies = []

        # 预热
        for _ in range(n_warmup):
            # preprocess
            tensor = torch.from_numpy(dummy_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            tensor = tensor.to(device)
            # 复制成 batch
            if bs > 1:
                tensor = tensor.expand(bs, -1, -1, -1).contiguous()
            # infer
            preds = model(tensor)
            # decode + nms
            res = model.decode(preds, conf_thresh=0.25)
            for b in range(bs):
                boxes, scores, _ = res[b]
                if boxes.numel():
                    nms(boxes, scores)

        # 正式测试
        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        for _ in range(n_iter):
            t0 = time.perf_counter()

            # ── preprocess ──
            tensor = torch.from_numpy(dummy_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            tensor = tensor.to(device)
            if bs > 1:
                tensor = tensor.expand(bs, -1, -1, -1).contiguous()

            # ── infer ──
            if device.type == 'cuda':
                start_event.record()
                preds = model(tensor)
                end_event.record()
                torch.cuda.synchronize()
                infer_ms = start_event.elapsed_time(end_event)
            else:
                t1 = time.perf_counter()
                preds = model(tensor)
                infer_ms = (time.perf_counter() - t1) * 1000

            # ── decode + nms ──
            t2 = time.perf_counter()
            res = model.decode(preds, conf_thresh=0.25)
            for b in range(bs):
                boxes, scores, _ = res[b]
                if boxes.numel():
                    nms(boxes, scores)
            post_ms = (time.perf_counter() - t2) * 1000

            # ── total ──
            total_ms = (time.perf_counter() - t0) * 1000
            latencies.append({
                'total_ms': total_ms,
                'infer_ms': infer_ms,
                'post_ms': post_ms,
            })

        # 统计
        totals = [l['total_ms'] for l in latencies]
        infers = [l['infer_ms'] for l in latencies]
        posts  = [l['post_ms'] for l in latencies]

        totals.sort()
        infers.sort()
        posts.sort()

        def stats(arr):
            return {
                'avg': float(np.mean(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'p50': arr[len(arr)//2],
                'p95': arr[int(len(arr)*0.95)],
                'p99': arr[int(len(arr)*0.99)],
            }

        r = {
            'batch_size': bs,
            'device': str(device),
            'total': stats(totals),
            'infer': stats(infers),
            'postproc': stats(posts),
            'fps': bs / (sum(totals)/len(totals)/1000),
        }
        results[bs] = r

        print(
            f'batch={bs:2d}  |  '
            f'{r["fps"]:>8.0f} FPS  |  '
            f'avg={r["total"]["avg"]:>7.2f}ms  '
            f'p50={r["total"]["p50"]:>7.2f}ms  '
            f'p95={r["total"]["p95"]:>7.2f}ms  '
            f'p99={r["total"]["p99"]:>7.2f}ms  |  '
            f'infer={r["infer"]["avg"]:>.2f}  '
            f'post={r["postproc"]["avg"]:>.2f}'
        )

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='YOLO-LAT 性能基准测试')
    parser.add_argument('--weights', default='latdet/runs/best.pt')
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8, 16, 32, 64])
    parser.add_argument('--cpu', action='store_true', help='测试CPU')
    parser.add_argument('--output', default=None, help='结果保存路径 (json)')
    args = parser.parse_args()

    devices = ['cpu'] if args.cpu else ['cuda', 'cpu']

    all_results = {}

    for device_name in devices:
        device = torch.device(device_name)
        if device_name == 'cuda' and not torch.cuda.is_available():
            print('CUDA 不可用, 跳过')
            continue

        ckpt = torch.load(args.weights, map_location=device)
        nc = ckpt.get('head.conv.1.weight', ckpt.get('head.weight')).shape[0] - 5
        model = YOLOLAT(nc=nc).to(device)
        model.load_state_dict(ckpt)
        model.eval()

        if device_name == 'cuda':
            # GPU 预热
            dummy = torch.randn(8, 3, args.imgsz, args.imgsz).cuda()
            for _ in range(50):
                model(dummy)

        results = benchmark(
            model, device, args.imgsz,
            args.batch_sizes,
            n_warmup=100, n_iter=500,
        )
        all_results[device_name] = results

    # 汇总
    print(f'\n{"="*50}')
    print('SUMMARY')
    print(f'{"="*50}')
    print(f'{"Device":<8} {"Batch":>5} {"FPS":>10} {"Avg":>8} {"P50":>8} {"P95":>8} {"P99":>8}')
    for dev, results in all_results.items():
        for bs, r in results.items():
            t = r['total']
            print(f'{dev:<8} {bs:>5} {r["fps"]:>10.0f} {t["avg"]:>8.2f}ms {t["p50"]:>8.2f}ms {t["p95"]:>8.2f}ms {t["p99"]:>8.2f}ms')

    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2, default=str))
        print(f'\n结果已保存: {args.output}')


if __name__ == '__main__':
    main()
