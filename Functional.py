import torch
import torch.nn.functional as F


def linear_kernels_attn_causal(phi, psi, V):
    C = torch.cumsum(psi, dim=1)
    S = torch.cumsum(
        torch.einsum("...ld,...lv->...ldv", psi, V), dim=1
    )  # (B, L, D+2, d_v)

    numerator = torch.einsum("...ld,...ldv->...lv", phi, S)
    denominator = torch.sum(phi * C, dim=-1, keepdim=True)

    Y = numerator / denominator.clamp(min=1e-8)
    return Y


def linear_kernels_attn(phi, psi, V):
    C = torch.sum(psi, dim=1)
    S = torch.sum(torch.einsum("...ld,...lv->...ldv", psi, V), dim=1)
    numerator = torch.einsum("...ld,...dv->...lv", phi, S)  # (..., L, d_v)
    denominator = torch.einsum("...ld,...d->...l", phi, C)  # (..., L)
    denominator = denominator.unsqueeze(-1)  # (..., L, 1)

    Y = numerator / denominator.clamp(min=1e-8)
    return Y


def linear_dot_causal(A, B, V):
    """
    归一化点积核：
    k =(p(A)·p(B) + 1)*0.5
    """
    A = torch.exp(A)
    B = torch.exp(B)

    phi = torch.cat([A, torch.ones_like(A[..., :1])], dim=-1)
    psi = torch.cat([B, torch.ones_like(B[..., :1])], dim=-1)

    return linear_kernels_attn_causal(phi, psi, V)


def linear_dot_modulated_causal(A, B, V):
    """
    保留模长的归一化点积核：
    k = (||A||^2+1) * (||B||^2+1) * (A_norm·B_norm + 1)
    """
    A_norm = F.normalize(A, p=2, dim=-1)  # (B, L, D)
    B_norm = F.normalize(B, p=2, dim=-1)

    scale_A = torch.norm(A**2, dim=-1, keepdim=True)
    scale_B = torch.norm(B**2, dim=-1, keepdim=True)

    phi = torch.cat([A_norm, torch.ones_like(A_norm[..., :1])], dim=-1)
    phi = phi * scale_A

    psi = torch.cat([B_norm, torch.ones_like(B_norm[..., :1])], dim=-1)
    psi = psi * scale_B

    return linear_kernels_attn_causal(phi, psi, V)


def linear_L2_attn_causal_vectorized(A, B, V):
    """
    A, B: (B, L, D)
    V:    (B, L, d_v)
    返回: Y: (B, L, d_v)
    注意力权重 = ||A_i + B_j||^2 / row_sum[i]
    """
    a_norm_sq = torch.sum(A**2, dim=-1, keepdim=True)  # (B, L, 1)
    b_norm_sq = torch.sum(B**2, dim=-1, keepdim=True)

    phi = torch.cat(
        [A, a_norm_sq, torch.ones_like(a_norm_sq)], dim=-1
    )  # φ(A) (B, L, D+2)
    psi = torch.cat(
        [2.0 * B, torch.ones_like(b_norm_sq), b_norm_sq], dim=-1
    )  # ψ(B) (B, L, D+2)

    return linear_kernels_attn_causal(phi, psi, V)


def linear_hadamard_attn_causal(A, B, V):
    """
    A, B: (B, L, D)
    V:    (B, L, d_v)
    Return: Y: (B, L, d_v)
    K = ||A_i * B_j
    """
    phi = torch.exp(A)
    psi = torch.exp(B)
    return linear_kernels_attn_causal(phi, psi, V)


def linear_hadamard_attn(A, B, V):
    """
    A, B: (B, L, D)
    V:    (B, L, d_v)
    Return: Y: (B, L, d_v)
    K = ||A_i * B_j
    """
    phi = torch.exp(A)
    psi = torch.exp(B)
    return linear_kernels_attn(phi, psi, V)
