"""
YOLO-LAT-ONNX: CPU优化的检测模型，可直接导出ONNX。

改动:
  1. ReLU 替代 SiLU (CPU更快)
  2. Conv2d 替代 DSC (depthwise在CPU上带宽受限)
  3. Hadamard attention 改用 bmm (替代 einsum，支持ONNX导出)
  4. decode 内嵌在 forward 中 → 直接输出 [cx, cy, w, h, score]
  5. 全卷积结构，分辨无关 (stride 16)
  6. BN融合工具 (推理时消除BN层)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── ONNX兼容的 Hadamard 线性注意力 ──────────────────────────────────

def linear_hadamard_attn_onnx(A, B, V):
    """
    Hadamard kernel线性注意力 (ONNX友好版).

    A, B: (B, L, D)  — query, key
    V:    (B, S, d_v) — value
    return: (B, L, d_v)

    用 bmm 替代 einsum，兼容 ONNX 导出。
    """
    phi = torch.exp(A)                     # (B, L, D)
    psi = torch.exp(B)                     # (B, S, D)

    C = psi.sum(dim=1, keepdim=False)       # (B, D)
    S = torch.bmm(psi.transpose(1, 2), V)  # (B, D, d_v)

    numerator = torch.bmm(phi, S)           # (B, L, d_v)
    denominator = torch.bmm(phi, C.unsqueeze(-1))  # (B, L, 1)

    return numerator / denominator.clamp(min=1e-8)


class HadamardAttention2D_ONNX(nn.Module):
    """2D Hadamard attention (ONNX可导出)."""
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** 0.5

        self.qkv = nn.Conv2d(d_model, d_model * 3, 1, bias=False)
        self.proj = nn.Conv2d(d_model, d_model, 1, bias=False)
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # (B, nh, hd, H*W) → (B, nh, L, hd)
        q = q.reshape(B, self.n_heads, self.head_dim, H * W).transpose(-1, -2)
        k = k.reshape(B, self.n_heads, self.head_dim, H * W).transpose(-1, -2)
        v = v.reshape(B, self.n_heads, self.head_dim, H * W).transpose(-1, -2)

        outs = []
        for h in range(self.n_heads):
            q_h = q[:, h] / self.scale
            k_h = k[:, h] / self.scale
            v_h = v[:, h]
            out_h = linear_hadamard_attn_onnx(q_h, k_h, v_h)
            outs.append(out_h)

        out = torch.stack(outs, dim=1)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        out = self.proj(out)
        return self.norm(x + out)


# ── Conv + BN + ReLU (CPU优化) ─────────────────────────────────────

class ConvBNR(nn.Module):
    """Conv2d + BN + ReLU — CPU优化的基础单元."""
    def __init__(self, c1, c2, k=3, s=1, g=1):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBN(nn.Module):
    """Conv2d + BN (无激活 — 用于head和projection)"""
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        return self.bn(self.conv(x))


# ── YOLO-LAT ONNX ───────────────────────────────────────────────────

class YOLOLAT_ONNX(nn.Module):
    """
    YOLO-LAT CPU优化版，可直接导出ONNX。

    输出: (B, L, 5)  — [cx, cy, w, h, score] 其中 L = ceil(H/16)*ceil(W/16)
    全卷积结构，支持任意 H,W (需 ≥ stride=16).
    """
    def __init__(self, nc=1, d_model=64, n_heads=4):
        super().__init__()
        self.nc = nc
        self.stride = 16
        self.n_out = 5 + nc  # raw: obj + xywh + cls(可能为空)

        # ── Backbone (全 Conv2d, ReLU, CPU优化) ──
        # 320 → 160 → 80 → 40 → 20
        self.stem = ConvBNR(3, 16, k=3, s=2)                     # s=2
        self.stage1 = ConvBNR(16, 32, k=3, s=2)                  # s=4
        self.stage2 = nn.Sequential(
            ConvBNR(32, 64, k=3, s=2),                           # s=8
            ConvBNR(64, 64, k=3, s=1),
        )
        self.stage3 = nn.Sequential(
            ConvBNR(64, self.stride * 8, k=3, s=2),              # s=16
            ConvBNR(self.stride * 8, self.stride * 8, k=3, s=1),
        )

        # ── Neck (Hadamard 注意力) ──
        C = self.stride * 8  # 128
        self.neck_proj = ConvBN(C, d_model, k=1)
        self.attention = HadamardAttention2D_ONNX(d_model, n_heads)
        self.neck_fuse = ConvBN(d_model, C, k=1)

        # ── Head ──
        self.head = nn.Conv2d(C, self.n_out, 1)

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

    def forward_raw(self, x):
        """返回原始head输出 (训练时用, 含loss所需的logits)."""
        B, _, H, W = x.shape
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        skip = x
        x = self.neck_proj(x)
        x = self.attention(x)
        x = self.neck_fuse(x)
        x = x + skip
        raw = self.head(x)  # (B, 5, h, w)
        return raw

    def forward(self, x):
        """输出直接解码的检测结果.

        Args:
            x: (B, 3, H, W)  float32, 无需固定尺寸

        Returns:
            out: (B, L, 5)  — [cx, cy, w, h, score], 归一化 [0,1]
                L = ceil(H/16) * ceil(W/16)
        """
        B, _, H, W = x.shape

        # ── Backbone ──
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)                     # (B, 128, H/16, W/16)

        # ── Neck ──
        skip = x
        x = self.neck_proj(x)
        x = self.attention(x)
        x = self.neck_fuse(x)
        x = x + skip

        # ── Head ──
        raw = self.head(x)                     # (B, 5+nc, h, w)
        _, _, h, w = raw.shape
        det = raw[:, :5]                       # 只取前5维用于decode (obj + xywh)

        # ── Decode (内嵌, ONNX兼容) ──
        obj = torch.sigmoid(det[:, 0:1])       # (B, 1, h, w)

        # 动态grid坐标 (ONNX arange 支持动态尺寸)
        xs = torch.arange(w, dtype=x.dtype, device=x.device).view(1, 1, 1, w)
        ys = torch.arange(h, dtype=x.dtype, device=x.device).view(1, 1, h, 1)

        xy = torch.sigmoid(det[:, 1:3])        # (B, 2, h, w)
        wh = torch.sigmoid(det[:, 3:5])        # (B, 2, h, w)

        cx = (xs + xy[:, 0:1]) / w             # 归一化 [0, 1]
        cy = (ys + xy[:, 1:2]) / h
        bw = (wh[:, 0:1] * 2) / w
        bh = (wh[:, 1:2] * 2) / h

        # 拼接并展平: (B, 5, h*w) → (B, L, 5)
        out = torch.cat([cx, cy, bw, bh, obj], dim=1)  # (B, 5, h, w)
        out = out.reshape(B, 5, -1).transpose(1, 2)  # (B, L, 5)
        return out


    def fuse_bn(self):
        """融合所有 BN 到 Conv，推理加速."""
        for m in self.modules():
            if isinstance(m, ConvBNR):
                self._fuse_conv_bn(m.conv, m.bn)
                m.bn = nn.Identity()
                m.relu = nn.ReLU(inplace=True)
            elif isinstance(m, ConvBN):
                self._fuse_conv_bn(m.conv, m.bn)
                m.bn = nn.Identity()
            elif isinstance(m, HadamardAttention2D_ONNX):
                self._fuse_conv_bn(m.qkv, m.qkv if hasattr(m.qkv, 'bias') else None)
                self._fuse_conv_bn(m.proj, m.proj if hasattr(m.proj, 'bias') else None)
                if hasattr(m.norm, 'running_mean'):  # BN
                    self._fuse_conv_bn(None, m.norm, inplace=True)  # skip, handled separately
        return self

    @staticmethod
    def _fuse_conv_bn(conv, bn):
        """融合 BN 参数到 Conv 权重."""
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


# ── Softmax Attention (经典注意力, 用于对比) ───────────────────────

class SoftmaxAttention2D(nn.Module):
    """
    标准softmax多头注意力 (使用F.scaled_dot_product_attention).

    Complexity: O(L² · d) — 随序列长度L二次增长.
    """
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Conv2d(d_model, d_model * 3, 1, bias=False)
        self.proj = nn.Conv2d(d_model, d_model, 1, bias=False)
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = q.reshape(B, self.n_heads, self.head_dim, H * W).transpose(-1, -2)
        k = k.reshape(B, self.n_heads, self.head_dim, H * W).transpose(-1, -2)
        v = v.reshape(B, self.n_heads, self.head_dim, H * W).transpose(-1, -2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        out = self.proj(out)
        return self.norm(x + out)


class YOLOLAT_Softmax(YOLOLAT_ONNX):
    """YOLO-LAT 但使用标准softmax注意力, 其余结构完全一致."""
    def __init__(self, nc=1, d_model=64, n_heads=4):
        super().__init__(nc, d_model, n_heads)
        self.attention = SoftmaxAttention2D(d_model, n_heads)


# ── 测试 ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time
    model = YOLOLAT_ONNX(nc=1)
    model.eval()

    # 不同分辨率测试
    for size in [224, 320, 416, 640]:
        x = torch.randn(1, 3, size, size)
        y = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f'Input: {size}x{size}  →  Output: {list(y.shape)}  |  Params: {params:,}')

    # CPU 基准
    device = torch.device('cpu')
    model = model.to(device)
    x = torch.randn(1, 3, 320, 320).to(device)
    for _ in range(50):
        model(x)
    t0 = time.perf_counter()
    for _ in range(200):
        model(x)
    t1 = time.perf_counter()
    print(f'\nCPU FPS (320x320): {200 / (t1 - t0):.0f}')
