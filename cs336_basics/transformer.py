from __future__ import annotations

from torch import Tensor

from .attention import multihead_self_attention_with_rope
from .nn_utils import embedding, linear, rmsnorm, swiglu


def transformer_block(
    in_features: Tensor,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
) -> Tensor:
    """单个 Transformer Block（Pre-Norm）。

    方法：RMSNorm → 多头注意力 → 残差；再 RMSNorm → SwiGLU FFN → 残差。
    关键变量：num_heads/d_ff 控制结构；weights 提供各子层参数。
    解决问题：在一层内融合注意力与前馈变换，提升表达能力。
    """
    _ = d_ff
    x = in_features
    # Pre-Norm + 残差：先归一化，再做注意力，再残差回加
    x_norm = rmsnorm(x, weights["ln1.weight"], eps=1e-5)
    attn_out = multihead_self_attention_with_rope(
        in_features=x_norm,
        num_heads=num_heads,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        theta=theta,
        max_seq_len=max_seq_len,
        token_positions=None,
    )
    x = x + attn_out

    # FFN 残差分支：RMSNorm + SwiGLU + 残差
    x_norm = rmsnorm(x, weights["ln2.weight"], eps=1e-5)
    ffn_out = swiglu(
        in_features=x_norm,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
    )
    x = x + ffn_out
    return x


def transformer_lm(
    in_indices: Tensor,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
) -> Tensor:
    """Transformer 语言模型前向计算。

    方法：token 嵌入 → 堆叠 num_layers 个 transformer_block → RMSNorm → LM head。
    关键变量：context_length 限制序列长度；rope_theta 控制 RoPE。
    解决问题：输出每个位置的词表 logits，用于语言建模训练/推理。
    """
    _ = vocab_size
    _ = d_model
    if in_indices.shape[-1] > context_length:
        raise ValueError("Input sequence length exceeds context_length.")

    # 词嵌入 + 多层 Transformer
    # in_indices: (batch, seq)，embedding 后为 (batch, seq, d_model)
    x = embedding(in_indices, weights["token_embeddings.weight"])
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."
        # 取出当前层的参数子集，去掉前缀方便索引
        layer_weights = {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
        x = transformer_block(
            in_features=x,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
        )
    x = rmsnorm(x, weights["ln_final.weight"], eps=1e-5)
    # 线性投影到词表维度，得到每个位置的 logits
    return linear(x, weights["lm_head.weight"])
