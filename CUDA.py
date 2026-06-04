"""
CUDA.py — Load and expose custom CUDA kernels for Graft linear attention.

On first import this module JIT-compiles cuops.cu via PyTorch's
cpp_extension machinery. Subsequent imports are cached.

Usage (drop-in replacement for Functional.py operators):
    from CUDA import linear_kernels_attn_causal, linear_kernels_attn

    Y = linear_kernels_attn_causal(phi, psi, V)   # causal / cumsum
    Y = linear_kernels_attn(phi, psi, V)           # non-causal / global sum
"""

import os
import torch
from torch.utils.cpp_extension import load as _load_ext


# Internal handle — loaded lazily so CUDA.py can be imported even on CPU-only
# machines, as long as no one actually calls the operators.
_cuops = None


def _get_cuops():
    """Lazy-load (JIT-compile) the CUDA extension on first call."""
    global _cuops
    if _cuops is not None:
        return _cuops

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(src_dir, "cuops.cu")

    if not os.path.exists(cu_path):
        raise FileNotFoundError(
            f"cuops.cu not found at {cu_path}. "
            f"Make sure the file exists in the same directory as CUDA.py."
        )

    # JIT-compile the extension
    _cuops = _load_ext(
        name="graft_cuops",
        sources=[cu_path],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-allow-unsupported-compiler",
        ],
        verbose=False,
    )
    return _cuops


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — drop-in replacements for Functional.py
# ═══════════════════════════════════════════════════════════════════════════════

def linear_kernels_attn_causal(
    phi: torch.Tensor, psi: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """
    Causal linear-kernel attention (fused CUDA).

    Implements:
        C = cumsum(psi, dim=1)                           # (…, L, D)
        S = cumsum(einsum("...ld,...lv->...ldv", psi, V), dim=1)  # (…, L, D, d_v)
        numerator   = einsum("...ld,...ldv->...lv", phi, S)
        denominator = sum(phi * C, dim=-1, keepdim=True)
        Y = numerator / clamp(denominator, min=1e-8)

    Args:
        phi: (…, L, D)  — left feature map
        psi: (…, L, D)  — right feature map
        V:   (…, L, d_v) — values
    Returns:
        Y: (…, L, d_v)
    """
    return _get_cuops().linear_kernels_attn_causal(
        phi.contiguous(), psi.contiguous(), V.contiguous()
    )


def linear_kernels_attn(
    phi: torch.Tensor, psi: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """
    Non-causal linear-kernel attention (CUDA, two-stage).

    Implements:
        C = sum(psi, dim=1)                               # (…, D)
        S = sum(einsum("...ld,...lv->...ldv", psi, V), dim=1)  # (…, D, d_v)
        numerator   = einsum("...ld,...dv->...lv", phi, S)
        denominator = einsum("...ld,...d->...l", phi, C).unsqueeze(-1)
        Y = numerator / clamp(denominator, min=1e-8)

    Args:
        phi: (…, L, D)   — left feature map
        psi: (…, L, D)   — right feature map
        V:   (…, L, d_v) — values
    Returns:
        Y: (…, L, d_v)
    """
    return _get_cuops().linear_kernels_attn(
        phi.contiguous(), psi.contiguous(), V.contiguous()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: build / pre-compile on import (optional)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Pre-compile the extension when run as a script
    print("Compiling Graft CUDA extension …")
    ops = _get_cuops()
    print("✓ Compilation successful.")
    print(f"  Available functions: {dir(ops)}")
