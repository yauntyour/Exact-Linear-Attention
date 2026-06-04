"""
YOLO-LAT detection & FPS benchmark.

Usage:
    python latdet/detect.py                          # benchmark FPS
    python latdet/detect.py --source data/images/0001.jpg  # single image
    python latdet/detect.py --source video.mp4             # video file
    python latdet/detect.py --weights latdet/runs/best.pt  # load trained weights
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latdet.model import YOLOLAT, nms


# ── Visualization ──────────────────────────────────────────────────────

def draw_boxes(img, boxes, scores, color=(0, 255, 0), thickness=2):
    """Draw normalized [cx,cy,w,h] boxes on image. img: (H,W,3) numpy uint8."""
    H, W = img.shape[:2]
    for box, score in zip(boxes, scores):
        cx, cy, w, h = box.tolist()
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, f'{score:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


# ── Inference ──────────────────────────────────────────────────────────

class Detector:
    """
    YOLO-LAT inference wrapper.

    Handles preprocessing, inference, decoding, and NMS.
    """
    def __init__(self, weights=None, device='cuda', conf=0.25, iou=0.5, nc=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf = conf
        self.iou = iou

        # 从 checkpoint 自动推断类别数
        if nc is None and weights and Path(weights).exists():
            ckpt = torch.load(weights, map_location='cpu')
            nc = ckpt.get('head.conv.1.weight', ckpt.get('head.weight')).shape[0] - 5
        elif nc is None:
            nc = 1

        self.model = YOLOLAT(nc=nc).to(self.device)
        if weights and Path(weights).exists():
            self.model.load_state_dict(torch.load(weights, map_location=self.device))
            print(f'Loaded weights from {weights}')
        self.model.eval()

        # Warm up
        dummy = torch.randn(1, 3, 320, 320).to(self.device)
        with torch.no_grad():
            for _ in range(30):
                self.model(dummy)
        print(f'Detector ready on {self.device}')

    @torch.no_grad()
    def __call__(self, img):
        """
        Args:
            img: (H,W,3) numpy uint8 BGR image (OpenCV default)

        Returns:
            boxes: (N, 4) [cx, cy, w, h] normalized
            scores: (N,)
        """
        H0, W0 = img.shape[:2]

        # Preprocess: resize to 320×320, normalize
        img_resized = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        # Inference
        preds = self.model(tensor)

        # Decode
        result = self.model.decode(preds, conf_thresh=self.conf)
        boxes, scores, cls_ids = result[0]

        if boxes.numel() == 0:
            return boxes, scores

        # NMS
        keep = nms(boxes, scores, iou_thresh=self.iou)
        return boxes[keep], scores[keep]

    @torch.no_grad()
    def benchmark(self, batch_size=1, n_warmup=50, n_iter=500):
        """FPS benchmark with synthetic data."""
        dummy = torch.randn(batch_size, 3, 320, 320).to(self.device)
        for _ in range(n_warmup):
            self.model(dummy)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            self.model(dummy)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        total = t1 - t0
        fps = n_iter / total
        print(f'\nBenchmark ({batch_size}x320x320):')
        print(f'  Total: {total:.2f}s for {n_iter} iterations')
        print(f'  FPS:   {fps:.1f} (batch={batch_size})')
        print(f'  Latency: {total/n_iter*1000:.2f} ms per batch')
        return fps


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='YOLO-LAT Detection')
    parser.add_argument('--source', type=str, default=None,
                        help='image path, video path, or 0 for webcam')
    parser.add_argument('--weights', type=str, default=None,
                        help='path to model weights')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--nc', type=int, default=1)
    parser.add_argument('--benchmark', action='store_true', default=True,
                        help='run FPS benchmark')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 8, 32],
                        help='batch sizes to benchmark')
    args = parser.parse_args()

    detector = Detector(args.weights, conf=args.conf, iou=args.iou, nc=args.nc)

    if args.source is not None:
        src = str(args.source)

        # Webcam
        if src == '0' or src.lower() == 'webcam':
            cap = cv2.VideoCapture(0)
        elif Path(src).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
            cap = cv2.VideoCapture(src)
            fps_in = cap.get(cv2.CAP_PROP_FPS)
            print(f'Video FPS: {fps_in:.1f}')
        else:
            # Single image
            img = cv2.imread(src)
            if img is None:
                print(f'Cannot read image: {src}')
                sys.exit(1)
            boxes, scores = detector(img)
            if boxes.numel() > 0:
                print(f'Detected {boxes.shape[0]} objects')
                img = draw_boxes(img, boxes, scores)
            cv2.imshow('YOLO-LAT', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            sys.exit(0)

        # Video / webcam loop
        if 'cap' in locals():
            fps_out = 0
            frame_count = 0
            t_start = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                boxes, scores = detector(frame)
                if boxes.numel() > 0:
                    frame = draw_boxes(frame, boxes, scores)

                # FPS overlay
                frame_count += 1
                elapsed = time.time() - t_start
                if elapsed > 1.0:
                    fps_out = frame_count / elapsed
                    frame_count = 0
                    t_start = time.time()

                cv2.putText(frame, f'FPS: {fps_out:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('YOLO-LAT', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    # FPS benchmark
    if args.benchmark:
        print('FPS Benchmark')
        print('=' * 40)
        for bs in args.batch_sizes:
            detector.benchmark(batch_size=bs)
