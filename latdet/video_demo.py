"""
YOLO-LAT 视频实时检测可视化

用法:
    python latdet/video_demo.py                      # 默认 res/000.avi
    python latdet/video_demo.py --source res/my_buff.avi
    python latdet/video_demo.py --source 0            # 摄像头
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
import cv2
import torch

from latdet.model import YOLOLAT, nms
from latdet.model_onnx import YOLOLAT_ONNX


def load_model(weights, device):
    """自动检测 checkpoint 格式，加载对应的模型类."""
    ckpt = torch.load(weights, map_location='cpu')

    # 判断是 ONNX 版还是原版
    is_onnx = 'head.weight' in ckpt and 'head.conv.1.weight' not in ckpt

    # 推断类别数
    for key in ['head.conv.1.weight', 'head.weight', 'module.head.conv.1.weight']:
        if key in ckpt:
            nc = ckpt[key].shape[0] - 5
            break
    else:
        nc = 1

    if is_onnx:
        # neck_proj.conv.weight shape = [d_model, C, 1, 1]
        d_model = ckpt['neck_proj.conv.weight'].shape[0]
        n_heads = 4
        model = YOLOLAT_ONNX(nc=nc, d_model=d_model, n_heads=n_heads)
    else:
        model = YOLOLAT(nc=nc)

    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    print(f"Model loaded | {sum(p.numel() for p in model.parameters()):,} params | {'ONNX' if is_onnx else 'Original'} arch")
    return model, is_onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="res/000.avi", help="视频路径或 0(摄像头)")
    parser.add_argument("--weights", default="latdet/runs_onnx/small_best.pt")
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--cpu", action="store_true", help="使用CPU推理")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    model, is_onnx = load_model(args.weights, device)

    # 打开视频
    src = args.source
    if src == "0" or src.lower() == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"无法打开视频: {src}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频: {w}x{h} | 按 q 退出")

    # 预热
    dummy = torch.randn(1, 3, args.imgsz, args.imgsz).to(device)
    for _ in range(20):
        model(dummy)

    # FPS 统计
    fps_avg = 0.0
    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 预处理 + 推理 ──
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(cv2.resize(img_rgb, (args.imgsz, args.imgsz)))
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        )
        tensor = tensor.to(device)

        with torch.no_grad():
            preds = model(tensor)
            if is_onnx:
                # ONNX 模型已内嵌 decode，直接输出 [cx,cy,w,h,score]
                dets = preds[0]
                mask = dets[:, 4] > args.conf
                dets = dets[mask]
                if dets.numel() > 0:
                    keep = nms(dets[:, :4], dets[:, 4], 0.5)
                    boxes, scores = dets[keep, :4], dets[keep, 4]
                else:
                    boxes, scores = torch.empty((0, 4), device=device), torch.empty(0, device=device)
            else:
                boxes, scores, _ = model.decode(preds, conf_thresh=args.conf)[0]
                keep = nms(boxes, scores)
                boxes, scores = boxes[keep], scores[keep]

        # ── FPS ──
        t_now = time.perf_counter()
        dt = t_now - t_prev
        t_prev = t_now
        alpha = 0.1
        fps_avg = fps_avg * (1 - alpha) + (1.0 / max(dt, 1e-6)) * alpha

        # ── 所有检测框（绿色）──
        for box, sc in zip(boxes, scores):
            cx, cy, bw, bh = box.tolist()
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{sc:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ── 置信度最高的框（红色粗框）──
        if boxes.numel() > 0:
            best = scores.argmax()
            cx, cy, bw, bh = boxes[best].tolist()
            sc = scores[best].item()
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                frame,
                f"BEST:{sc:.2f}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # ── 信息 ──
        cv2.putText(
            frame,
            f"YOLO-LAT | {fps_avg:.0f} FPS | det={len(boxes)} | q=exit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLO-LAT Detection (GREEN=all, RED=best)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
