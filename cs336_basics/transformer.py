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
    _ = d_ff
    x = in_features
    # Pre-Norm + 残差
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

    # FFN 残差分支
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
    _ = vocab_size
    _ = d_model
    if in_indices.shape[-1] > context_length:
        raise ValueError("Input sequence length exceeds context_length.")

    # 词嵌入 + 多层 Transformer
    x = embedding(in_indices, weights["token_embeddings.weight"])
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."
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
    # 线性投影到词表维度
    return linear(x, weights["lm_head.weight"])
