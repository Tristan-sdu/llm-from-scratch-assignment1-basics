from __future__ import annotations

import math
import torch
from torch import Tensor

from .nn_utils import linear, softmax


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    # 标准缩放点积注意力
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # mask 为 False 的位置置为 -inf，确保 softmax 后为 0
        mask = mask.to(dtype=torch.bool, device=scores.device)
        neg_inf = torch.finfo(scores.dtype).min
        scores = torch.where(mask, scores, neg_inf)
    attn = softmax(scores, dim=-1)
    return attn @ V


def _prepare_positions(token_positions: Tensor, target_dim: int) -> Tensor:
    # 将位置张量扩展到与输入张量前缀维度兼容
    pos = token_positions
    if pos.dim() == 1:
        pos = pos.view(1, -1)
    if pos.dim() == 2 and target_dim > 2:
        while pos.dim() < target_dim:
            pos = pos.unsqueeze(1)
    elif pos.dim() < target_dim:
        while pos.dim() < target_dim:
            pos = pos.unsqueeze(0)
    return pos


def rope(
    in_query_or_key: Tensor, theta: float, token_positions: Tensor, max_seq_len: int | None = None
) -> Tensor:
    # RoPE：对偶数/奇数维做旋转
    _ = max_seq_len
    d_k = in_query_or_key.shape[-1]
    if d_k % 2 != 0:
        raise ValueError("RoPE requires an even embedding dimension.")

    pos = _prepare_positions(token_positions.to(in_query_or_key.device), in_query_or_key.dim() - 1)
    pos = pos.to(dtype=in_query_or_key.dtype)

    half = d_k // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, device=in_query_or_key.device, dtype=in_query_or_key.dtype) * 2 / d_k)
    )
    freqs = pos.unsqueeze(-1) * inv_freq
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    x_even = in_query_or_key[..., 0::2]
    x_odd = in_query_or_key[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.stack((out_even, out_odd), dim=-1)
    return out.reshape(in_query_or_key.shape)


def _split_heads(x: Tensor, num_heads: int) -> Tensor:
    # (batch, seq, d_model) -> (batch, heads, seq, d_head)
    d_model = x.shape[-1]
    d_head = d_model // num_heads
    return x.view(*x.shape[:-1], num_heads, d_head).transpose(-3, -2)


def _merge_heads(x: Tensor) -> Tensor:
    # (batch, heads, seq, d_head) -> (batch, seq, d_model)
    x = x.transpose(-3, -2)
    new_shape = (*x.shape[:-2], x.shape[-2] * x.shape[-1])
    return x.reshape(new_shape)


def multihead_self_attention(
    in_features: Tensor,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
) -> Tensor:
    # 将 QKV 投影打包为一次批处理的多头注意力
    q = linear(in_features, q_proj_weight)
    k = linear(in_features, k_proj_weight)
    v = linear(in_features, v_proj_weight)

    q = _split_heads(q, num_heads)
    k = _split_heads(k, num_heads)
    v = _split_heads(v, num_heads)

    seq_len = in_features.shape[-2]
    # 语言模型因果 mask
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))
    mask = mask.view(1, 1, seq_len, seq_len)
    attn = scaled_dot_product_attention(q, k, v, mask=mask)
    merged = _merge_heads(attn)
    return linear(merged, o_proj_weight)


def multihead_self_attention_with_rope(
    in_features: Tensor,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    theta: float,
    max_seq_len: int,
    token_positions: Tensor | None = None,
) -> Tensor:
    seq_len = in_features.shape[-2]
    if token_positions is None:
        # 默认位置为 [0..seq_len-1]
        token_positions = torch.arange(seq_len, device=in_features.device)

    q = linear(in_features, q_proj_weight)
    k = linear(in_features, k_proj_weight)
    v = linear(in_features, v_proj_weight)

    q = _split_heads(q, num_heads)
    k = _split_heads(k, num_heads)
    v = _split_heads(v, num_heads)

    q = rope(q, theta=theta, token_positions=token_positions, max_seq_len=max_seq_len)
    k = rope(k, theta=theta, token_positions=token_positions, max_seq_len=max_seq_len)

    # RoPE 版本同样使用因果 mask
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))
    mask = mask.view(1, 1, seq_len, seq_len)
    attn = scaled_dot_product_attention(q, k, v, mask=mask)
    merged = _merge_heads(attn)
    return linear(merged, o_proj_weight)
