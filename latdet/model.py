"""
YOLO-LAT: Lightweight object detector with Hadamard linear attention.

Architecture:
- Backbone: depthwise separable CNN, 3 stages (320 → stride 16)
- Neck: Hadamard linear attention (linear_hadamard_attn from Functional.py)
- Head: lightweight conv detection head (obj + xywh + cls)

Params: ~200K  |  FLOPS: ~0.5G  |  Input: 3×320×320
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Functional import linear_hadamard_attn

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Basic building blocks ──────────────────────────────────────────────

class Conv(nn.Module):
    """Conv2d + BN + SiLU (standard YOLO-style conv block)"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSC(nn.Module):
    """Depthwise Separable Conv: DWConv3×3 + PWConv1×1"""
    def __init__(self, c1, c2, s=1):
        super().__init__()
        self.dw = Conv(c1, c1, k=3, s=s, g=c1)
        self.pw = Conv(c1, c2, k=1)

    def forward(self, x):
        return self.pw(self.dw(x))


# ── Hadamard Linear Attention (non-causal, bidirectional) ─────────────

class HadamardAttention2D(nn.Module):
    """
    Hadamard kernel linear attention for 2D feature maps.

    Uses linear_hadamard_attn (exp feature map) from Functional.py:
      k(Q_i, K_j) ≈ sum_d exp(Q_i_d) · exp(K_j_d)

    Complexity: O(B·L·d²) instead of O(B·L²)
    For 20×20 grid (L=400, d=64): ~1.6M ops vs 20M for softmax.
    """
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** 0.5  # temperature for numerical stability

        self.qkv = nn.Conv2d(d_model, d_model * 3, 1, bias=False)
        self.proj = nn.Conv2d(d_model, d_model, 1, bias=False)
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1×1 conv to project Q, K, V
        qkv = self.qkv(x)                           # (B, 3C, H, W)
        q, k, v = torch.chunk(qkv, 3, dim=1)        # each (B, C, H, W)

        # Reshape to per-head: (B, nh, L, hd)
        q = q.reshape(B, self.n_heads, self.head_dim, H * W).transpose(-1, -2)
        k = k.reshape(B, self.n_heads, self.head_dim, H * W).transpose(-1, -2)
        v = v.reshape(B, self.n_heads, self.head_dim, H * W).transpose(-1, -2)

        # Per-head Hadamard attention (non-causal)
        outs = []
        for h in range(self.n_heads):
            q_h = q[:, h] / self.scale          # (B, L, hd)  ← temperature for stable exp
            k_h = k[:, h] / self.scale
            v_h = v[:, h]                       # (B, L, hd)
            out_h = linear_hadamard_attn(q_h, k_h, v_h)   # (B, L, hd)
            outs.append(out_h)

        out = torch.stack(outs, dim=1)          # (B, nh, L, hd)
        out = out.transpose(-1, -2).reshape(B, C, H, W)  # (B, C, H, W)
        out = self.proj(out)

        # Residual + BN
        return self.norm(x + out)


# ── Detection head ─────────────────────────────────────────────────────

class DetectHead(nn.Module):
    """Single-scale detection head: lightweight conv → per-cell predictions"""
    def __init__(self, d_model, nc):
        super().__init__()
        self.nc = nc
        n_out = 5 + nc  # obj + xywh + class scores

        self.conv = nn.Sequential(
            Conv(d_model, d_model, k=3),
            nn.Conv2d(d_model, n_out, 1),
        )

    def forward(self, x):
        return self.conv(x)


# ── YOLO-LAT model ─────────────────────────────────────────────────────

class YOLOLAT(nn.Module):
    """
    YOLO-LAT: Lightweight object detector with Hadamard linear attention.

    Backbone:  3-stage depthwise separable CNN (320 → stride 16)
    Neck:      Hadamard linear attention (global receptive field, linear complexity)
    Head:      1-stage conv detection head

    Args:
        nc: number of detection classes
        d_model: feature dimension at neck/head
        n_heads: number of attention heads

    Shape:
        Input:  (B, 3, H, W) — recommended 320×320
        Output: (B, 5+nc, H/16, W/16) — per-cell predictions
    """
    def __init__(self, nc=1, d_model=64, n_heads=4):
        super().__init__()
        self.nc = nc
        self.d_model = d_model
        self.stride = 16
        self.n_out = 5 + nc

        # ── Backbone ──
        self.stem = Conv(3, 16, k=3, s=2)                   # 320 → 160
        self.stage1 = DSC(16, 32, s=2)                      # 160 → 80
        self.stage2 = DSC(32, 64, s=2)                      #  80 → 40
        self.stage3 = nn.Sequential(
            DSC(64, d_model * 2, s=2),                      #  40 → 20
            DSC(d_model * 2, d_model * 2, s=1),
        )

        # ── Neck: project to d_model + Hadamard attention ──
        self.neck_proj = Conv(d_model * 2, d_model, k=1)
        self.attention = HadamardAttention2D(d_model, n_heads)
        self.neck_fuse = Conv(d_model, d_model * 2, k=1)

        # ── Detection head ──
        self.head = DetectHead(d_model * 2, nc)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Backbone
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)              # (B, 128, H/16, W/16)

        # Neck: Hadamard attention
        identity = x
        x = self.neck_proj(x)           # (B, d_model, H/16, W/16)
        x = self.attention(x)           # global context
        x = self.neck_fuse(x)           # (B, 128, H/16, W/16)
        x = x + identity                # residual

        # Head
        out = self.head(x)              # (B, 5+nc, H/16, W/16)
        return out

    def decode(self, preds, conf_thresh=0.25):
        """
        Decode model predictions to normalized boxes.

        Args:
            preds: (B, 5+nc, H, W) raw model output
            conf_thresh: confidence threshold

        Returns:
            boxes:  (B, N, 4)  — [cx, cy, w, h] normalized [0, 1]
            scores: (B, N)
            cls_ids: (B, N) or None if nc=0
        """
        B, C, H, W = preds.shape
        device = preds.device

        # Grid
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij',
        )
        xs = xs.float().view(1, 1, H, W)
        ys = ys.float().view(1, 1, H, W)

        obj = preds[:, 0:1].sigmoid()               # (B, 1, H, W)
        xy = preds[:, 1:3].sigmoid()                # (B, 2, H, W)
        wh = preds[:, 3:5].sigmoid()                # (B, 2, H, W)

        # Decode to normalized coordinates [0, 1]
        cx = (xs + xy[:, 0:1]) / W
        cy = (ys + xy[:, 1:2]) / H
        w = (wh[:, 0:1] * 2) / W
        h = (wh[:, 1:2] * 2) / H

        boxes = torch.cat([cx, cy, w, h], dim=1)    # (B, 4, H, W)
        boxes = boxes.reshape(B, 4, -1).transpose(1, 2)  # (B, L, 4)
        scores = obj.reshape(B, 1, -1).squeeze(1)   # (B, L)

        if self.nc > 0:
            cls = preds[:, 5:].softmax(dim=1)       # (B, nc, H, W)
            cls_scores, cls_ids = cls.max(dim=1)    # (B, H, W)
            cls_scores = cls_scores.reshape(B, -1)
            cls_ids = cls_ids.reshape(B, -1)
            scores = scores * cls_scores

            # Filter by threshold
            mask = scores > conf_thresh
            result = []
            for b in range(B):
                result.append((
                    boxes[b][mask[b]],
                    scores[b][mask[b]],
                    cls_ids[b][mask[b]],
                ))
            return result

        return [(boxes[b], scores[b], None) for b in range(B)]


# ── Utilities ──────────────────────────────────────────────────────────

def compute_iou(boxes1, boxes2, mode='ciou'):
    """
    Compute (C)IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) [cx, cy, w, h] normalized
        boxes2: (N, 4)
        mode: 'iou' or 'ciou'

    Returns:
        iou: (N,)
    """
    # Convert to corner format
    x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    y2 = boxes1[:, 1] + boxes1[:, 3] / 2

    x1g = boxes2[:, 0] - boxes2[:, 2] / 2
    y1g = boxes2[:, 1] - boxes2[:, 3] / 2
    x2g = boxes2[:, 0] + boxes2[:, 2] / 2
    y2g = boxes2[:, 1] + boxes2[:, 3] / 2

    # Intersection
    ix1 = torch.max(x1, x1g)
    iy1 = torch.max(y1, y1g)
    ix2 = torch.min(x2, x2g)
    iy2 = torch.min(y2, y2g)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    # Union
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-7)

    if mode == 'iou':
        return iou

    # CIoU: add center distance + aspect ratio penalty
    enclose_x1 = torch.min(x1, x1g)
    enclose_y1 = torch.min(y1, y1g)
    enclose_x2 = torch.max(x2, x2g)
    enclose_y2 = torch.max(y2, y2g)
    enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    center_dist = (boxes1[:, 0] - boxes2[:, 0]) ** 2 + (boxes1[:, 1] - boxes2[:, 1]) ** 2

    v = (4 / math.pi ** 2) * (
        torch.atan(boxes2[:, 2] / boxes2[:, 3].clamp(min=1e-7)) -
        torch.atan(boxes1[:, 2] / boxes1[:, 3].clamp(min=1e-7))
    ) ** 2
    alpha = v / (1 - iou + v).clamp(min=1e-7)

    ciou = iou - center_dist / enclose_diag.clamp(min=1e-7) - alpha * v
    return ciou


def nms(boxes, scores, iou_thresh=0.5):
    """
    Non-maximum suppression.

    Args:
        boxes: (N, 4) [cx, cy, w, h] normalized
        scores: (N,)
        iou_thresh: IoU threshold

    Returns:
        keep: indices of kept boxes
    """
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    # Convert to corner format
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = boxes[:, 2] * boxes[:, 3]

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())

        if order.numel() == 1:
            break

        # IoU with remaining
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter).clamp(min=1e-7)

        mask = iou <= iou_thresh
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def model_size(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Simple test ────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = YOLOLAT(nc=1, d_model=64, n_heads=4)
    x = torch.randn(2, 3, 320, 320)
    y = model(x)
    print(f'Input:  {x.shape}')
    print(f'Output: {y.shape}')
    print(f'Params: {model_size(model):,}')

    # Warm-up + FPS estimate
    import time
    model.eval()
    with torch.no_grad():
        for _ in range(50):
            model(x)
        t0 = time.perf_counter()
        for _ in range(200):
            model(x)
        t1 = time.perf_counter()
    print(f'FPS: {200 / (t1 - t0):.1f} (batch=2, {x.shape[2]}×{x.shape[3]})')
