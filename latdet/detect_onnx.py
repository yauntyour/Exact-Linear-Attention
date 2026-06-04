"""
YOLO-LAT ONNX Runtime 推理 (CPU, 无PyTorch依赖)

模型输出: (B, L, 5) — [cx, cy, w, h, score] 归一化 [0,1]
分辨率无关: 任意 H,W ≥ 16

用法:
    python latdet/detect_onnx.py                                          # 视频
    python latdet/detect_onnx.py --image data/images/frame_0003.jpg       # 图片
    python latdet/detect_onnx.py --benchmark                              # 基准
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

try:
    import onnxruntime as ort
except ImportError:
    print('需要 onnxruntime: pip install onnxruntime')
    sys.exit(1)


class YOLOLAT_ONNX:
    """ONNX Runtime 封装 — 单帧推理"""

    def __init__(self, onnx_path='latdet/runs_onnx/yololat.onnx', conf=0.25):
        self.session = ort.InferenceSession(
            onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.conf = conf

        # 预热
        dummy = np.random.randn(1, 3, 320, 320).astype(np.float32)
        for _ in range(10):
            self.session.run(None, {self.input_name: dummy})
        print(f'ONNX loaded: {Path(onnx_path).name}  ({Path(onnx_path).stat().st_size/1024:.0f} KB)')

    def __call__(self, img, conf=None):
        """
        Args:
            img: (H, W, 3) BGR uint8 (OpenCV默认格式)

        Returns:
            boxes:  (N, 4)  [cx, cy, w, h] 归一化
            scores: (N,)
        """
        H, W = img.shape[:2]
        conf = conf or self.conf

        # Preprocess
        resized = cv2.resize(img, (320, 320))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0

        # Infer
        outs = self.session.run(None, {self.input_name: tensor})
        dets = outs[0][0]  # (L, 5)

        # Filter
        mask = dets[:, 4] > conf
        dets = dets[mask]

        if len(dets) == 0:
            return np.empty((0, 4)), np.empty(0)

        boxes = dets[:, :4]
        scores = dets[:, 4]

        # NMS (简单实现)
        keep = self._nms(boxes, scores)
        return boxes[keep], scores[keep]

    @staticmethod
    def _nms(boxes, scores, iou_thresh=0.5):
        """NMS — 纯numpy实现."""
        if len(boxes) == 0:
            return np.array([], dtype=int)

        # cx,cy,w,h → x1,y1,x2,y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        areas = boxes[:, 2] * boxes[:, 3]

        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
            order = order[1:][iou <= iou_thresh]
        return np.array(keep)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='YOLO-LAT ONNX Runtime')
    parser.add_argument('--onnx', default='latdet/runs_onnx/yololat.onnx')
    parser.add_argument('--source', default='res/000.avi', help='图片/视频路径, 0=摄像头')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()

    detector = YOLOLAT_ONNX(args.onnx, conf=args.conf)

    if args.benchmark:
        print('\nBenchmark (CPU, ONNX Runtime):')
        for size in [(320, 320), (416, 416), (640, 480)]:
            dummy = np.random.randn(size[1], size[0], 3).astype(np.uint8)
            for _ in range(30):
                detector(dummy)
            t0 = time.perf_counter()
            n = 500
            for _ in range(n):
                detector(dummy)
            t1 = time.perf_counter()
            avg = (t1 - t0) / n * 1000
            print(f'  {size[0]}x{size[1]}:  {avg:.2f}ms  ({1000/avg:.0f} FPS)')
        return

    # 视频 / 摄像头
    src = args.source
    if src == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f'Cannot open: {src}')
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Video: {w}x{h}  |  q=exit')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        boxes, scores = detector(frame)
        t1 = time.perf_counter()
        fps = 1.0 / (t1 - t0 + 1e-6)

        for box, sc in zip(boxes, scores):
            cx, cy, bw, bh = box
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{sc:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if len(boxes) > 0:
            best = scores.argmax()
            cx, cy, bw, bh = boxes[best]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f'BEST:{scores[best]:.2f}', (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, f'ONNX CPU | {fps:.0f} FPS | det={len(boxes)} | q=exit',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('YOLO-LAT ONNX (CPU)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
