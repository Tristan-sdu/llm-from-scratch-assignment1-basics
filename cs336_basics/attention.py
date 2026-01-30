from __future__ import annotations

import math
import torch
from torch import Tensor

from .nn_utils import linear, softmax


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    """缩放点积注意力。

    方法：计算 QK^T/√d_k 得分，按 mask 置 -inf 后做 softmax，再与 V 加权求和。
    关键变量：Q/K/V 为注意力输入；mask 为可见性矩阵；d_k 为每头维度。
    解决问题：在序列中按相似度聚合上下文信息，支持因果/padding mask。
    """
    # 标准缩放点积注意力
    # 形状约定：Q/K/V = (..., seq_len, d_k)，返回 (..., seq_len, d_k)
    d_k = Q.shape[-1]
    # 相似度分数：QK^T / sqrt(d_k)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # mask 为 False 的位置置为 -inf，确保 softmax 后为 0
        mask = mask.to(dtype=torch.bool, device=scores.device)
        neg_inf = torch.finfo(scores.dtype).min
        scores = torch.where(mask, scores, neg_inf)
    # 对最后一维做 softmax，得到注意力权重
    attn = softmax(scores, dim=-1)
    # 权重与 V 相乘得到加权和
    return attn @ V


def _prepare_positions(token_positions: Tensor, target_dim: int) -> Tensor:
    """对位置张量做维度对齐。

    方法：根据 target_dim 反复 unsqueeze，确保可与输入张量前缀维度广播。
    关键变量：token_positions 为位置索引；target_dim 为目标维度数。
    解决问题：为 RoPE 的位置广播做形状准备。
    """
    # 将位置张量扩展到与输入张量前缀维度兼容
    # 支持 token_positions 为 (seq,) 或 (batch, seq) 等常见形状
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
    """RoPE 旋转位置编码。

    方法：将偶/奇维成对旋转，等价于乘以 e^{i*freq*pos} 的复数旋转。
    关键变量：theta 为基频；token_positions 为位置；d_k 为头维度。
    解决问题：向 Q/K 注入相对位置信息，便于外推更长序列。
    """
    # RoPE：对偶数/奇数维做旋转
    # 将每对 (x_even, x_odd) 按位置编码旋转，等价于在复数平面乘以 e^{i*theta}
    _ = max_seq_len
    d_k = in_query_or_key.shape[-1]
    if d_k % 2 != 0:
        raise ValueError("RoPE requires an even embedding dimension.")

    # 位置张量对齐到输入前缀维度，便于广播
    pos = _prepare_positions(token_positions.to(in_query_or_key.device), in_query_or_key.dim() - 1)
    pos = pos.to(dtype=in_query_or_key.dtype)

    half = d_k // 2
    # 频率项：theta^{-2i/d_k}，与论文中的 10000^{-2i/d_k} 同形
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, device=in_query_or_key.device, dtype=in_query_or_key.dtype) * 2 / d_k)
    )
    # 每个位置与频率相乘，得到旋转角度
    freqs = pos.unsqueeze(-1) * inv_freq
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    # 将偶数/奇数维拆开旋转
    x_even = in_query_or_key[..., 0::2]
    x_odd = in_query_or_key[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.stack((out_even, out_odd), dim=-1)
    return out.reshape(in_query_or_key.shape)


def _split_heads(x: Tensor, num_heads: int) -> Tensor:
    """将最后一维拆分为多头。

    方法：view 分割 d_model -> (heads, d_head)，再 transpose 交换维度。
    关键变量：num_heads 为头数；d_head=d_model/num_heads。
    解决问题：为多头并行注意力计算准备形状。
    """
    # (batch, seq, d_model) -> (batch, heads, seq, d_head)
    # 先 reshape 再交换维度，避免显式 copy
    d_model = x.shape[-1]
    d_head = d_model // num_heads
    return x.view(*x.shape[:-1], num_heads, d_head).transpose(-3, -2)


def _merge_heads(x: Tensor) -> Tensor:
    """合并多头回到 d_model。

    方法：transpose 回原顺序，再 reshape 合并 heads 与 d_head。
    关键变量：x 形状为 (batch, heads, seq, d_head)。
    解决问题：将多头注意力结果回到单一表示空间。
    """
    # (batch, heads, seq, d_head) -> (batch, seq, d_model)
    # 与 _split_heads 的逆操作
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
    """标准多头自注意力。

    方法：线性投影得到 Q/K/V，按头拆分，因果 mask 下做注意力，再输出投影。
    关键变量：投影权重 q/k/v/o；num_heads 控制头数。
    解决问题：在自回归语言模型中建模序列内部依赖。
    """
    # 将 QKV 投影打包为一次批处理的多头注意力
    # 输入形状：(batch, seq, d_model)
    q = linear(in_features, q_proj_weight)
    k = linear(in_features, k_proj_weight)
    v = linear(in_features, v_proj_weight)

    # 拆分多头，形状变为 (batch, heads, seq, d_head)
    q = _split_heads(q, num_heads)
    k = _split_heads(k, num_heads)
    v = _split_heads(v, num_heads)

    seq_len = in_features.shape[-2]
    # 语言模型因果 mask：只允许关注自己及之前的位置
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))
    mask = mask.view(1, 1, seq_len, seq_len)
    attn = scaled_dot_product_attention(q, k, v, mask=mask)
    # 合并多头并做输出投影
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
    """带 RoPE 的多头自注意力。

    方法：与标准 MHA 类似，但对 Q/K 注入 RoPE 位置旋转。
    关键变量：theta 与 token_positions 控制位置编码；max_seq_len 用于接口一致性。
    解决问题：在保持因果注意力的同时提供相对位置信息。
    """
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

    # 对 Q/K 注入 RoPE 位置信息
    q = rope(q, theta=theta, token_positions=token_positions, max_seq_len=max_seq_len)
    k = rope(k, theta=theta, token_positions=token_positions, max_seq_len=max_seq_len)

    # RoPE 版本同样使用因果 mask
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))
    mask = mask.view(1, 1, seq_len, seq_len)
    attn = scaled_dot_product_attention(q, k, v, mask=mask)
    merged = _merge_heads(attn)
    return linear(merged, o_proj_weight)
