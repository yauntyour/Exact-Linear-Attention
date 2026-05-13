import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# 0. 线性门控FFN (SwiGLU)
# -------------------------------
class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or in_dim
        self.linear = nn.Linear(in_dim, 2 * out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        gate, value = x.chunk(2, dim=-1)
        return self.out_proj(F.silu(gate) * value)


# -------------------------------
# 1. 旋转位置编码 (RoPE)
# -------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.register_buffer(
            "inv_freq", 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        )

    def _get_freqs_cis(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, q, k, start_pos=0):
        Lq, Lk = q.size(2), k.size(2)
        freqs_cis = self._get_freqs_cis(max(Lq, Lk) + start_pos, q.device)
        q_freqs = freqs_cis[start_pos : start_pos + Lq]
        k_freqs = freqs_cis[start_pos : start_pos + Lk]
        q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        q_out = torch.view_as_real(q_ * q_freqs.unsqueeze(0).unsqueeze(1)).flatten(3)
        k_out = torch.view_as_real(k_ * k_freqs.unsqueeze(0).unsqueeze(1)).flatten(3)
        return q_out.type_as(q), k_out.type_as(k)


# -------------------------------
# 2. RoPE多头注意力
# -------------------------------
class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nheads):
        super().__init__()
        assert d_model % nheads == 0
        self.d_model = d_model
        self.nExperts = nheads
        self.head_dim = d_model // nheads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, query, key, value):
        B, Lq, _ = query.shape
        Lk = key.size(1)

        q = self.q_proj(query).view(B, Lq, self.nExperts, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, self.nExperts, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, self.nExperts, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.out_proj(attn_out)


# -------------------------------
# 3. FFN
# -------------------------------
class MoE(nn.Module):
    """
    Mix of Experts (MoE) 激活的 FFN
    参数:
        d_model   : 输入/输出特征维度
        nExperts     : 专家数量
        top_k     : 每个 token 激活的专家数量
        aux_coef  : 负载均衡损失系数 (0 表示禁用)
    """

    def __init__(self, d_model, nExperts, top_k=2, aux_coef=1e-2):
        super().__init__()
        self.d_model = d_model
        self.nExperts = nExperts
        self.top_k = top_k
        self.aux_coef = aux_coef

        # 门控：x -> 专家 logits
        self.gate = nn.Linear(d_model, nExperts)

        # SwiGLU 专家集合
        self.experts = nn.ModuleList(
            [SwiGLU(in_dim=d_model, out_dim=d_model) for _ in range(nExperts)]
        )

    def forward(self, x):
        """
        输入:
            x            : [B, S, d_model]
        输出:
            x            : 融入专家输出后的特征 (已加残差)
            balance_loss : 负载均衡辅助损失 (标量)
        """
        residual = x
        balance_loss = torch.tensor(0.0, device=x.device)

        # 1) 门控分数
        gate_logits = self.gate(x)  # [B, S, nExperts]
        gate_scores = torch.softmax(gate_logits, dim=-1)

        # 2) top-k 选择并归一化权重
        topk_weights, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 3) 专家计算 + 加权求和
        out = torch.zeros_like(x)
        for e in range(self.nExperts):
            mask_e = topk_indices == e  # [B, S, top_k]
            token_mask = mask_e.any(dim=-1)  # [B, S] bool
            if token_mask.any():
                selected_x = x[token_mask]  # [N, D]
                expert_out = self.experts[e](selected_x)  # [N, D]
                weight_e = topk_weights[mask_e]  # [N]
                out[token_mask] += (expert_out) * weight_e.unsqueeze(-1)
        x = residual + out

        if self.training:
            # 平均路由概率 g_i
            g = gate_scores.mean(dim=[0, 1])  # [nExperts]
            # 实际分配比例 f_i
            one_hot = F.one_hot(
                topk_indices.squeeze(-1), num_classes=self.nExperts
            ).float()  # [B, S, nExperts]
            f = one_hot.mean(dim=[0, 1])  # [nExperts]
            balance_loss = self.nExperts * torch.sum(f * g) * 0.01
            return x, balance_loss
        else:
            return x, 0


class LobeLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)
        self.Q = nn.Linear(d_model, d_model, bias=False)

    def forward(self, dx):
        q = self.Q(dx)
        k = self.K(dx)
        v = self.V(dx)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


# -------------------------------
# 4. Decoder 层
# -------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, nExperts):
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(d_model, n_heads)
        self.norm = nn.RMSNorm(d_model)
        self.MoE = MoE(d_model, nExperts)
        self.lob = LobeLayer(d_model)

    def forward(self, x):
        x_norm = self.norm(x)
        attn = self.self_attn(query=x_norm, key=x_norm, value=x_norm)

        ffn_out, aux_loss = self.MoE(attn)
        lob_out = self.lob(ffn_out)
        return x + ffn_out + lob_out, aux_loss


class DecoderLayerSTD(nn.Module):
    def __init__(self, d_model, n_heads, nExperts):
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.RMSNorm(d_model)
        self.MoE = MoE(d_model, nExperts)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x):
        x_norm = self.norm1(x)
        x = x + self.self_attn(query=x_norm, key=x_norm, value=x_norm)
        ffn_out, aux_loss = self.MoE(self.norm2(x))
        x = x + ffn_out
        return x, aux_loss


# -------------------------------
# 5. 完整模型
# -------------------------------
class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        nExperts=12,
        num_layers=12,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, nExperts) for _ in range(num_layers)]
        )

    def forward(self, input_ids):
        x = self.token_embed(input_ids)
        balance_loss = 0
        for layer in self.decoder_layers:
            x, aux_loss = layer(x)
            balance_loss += aux_loss

        logits = F.linear(x, self.token_embed.weight)
        return logits, balance_loss


# -------------------------------
# 训练损失函数
# -------------------------------
class loss_tools:
    def cross_loss(model, batch):
        x = batch[:, :-1]
        y = batch[:, 1:]

        logits, aux_loss = model(x)

        loss_clm = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        return loss_clm, aux_loss


model_name = "gpt"
