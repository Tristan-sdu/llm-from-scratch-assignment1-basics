from __future__ import annotations

import torch
from torch import Tensor


def linear(in_features: Tensor, weights: Tensor) -> Tensor:
    # 线性层：按 PyTorch Linear 的权重排布 (out, in)
    return in_features @ weights.t()


def embedding(token_ids: Tensor, weights: Tensor) -> Tensor:
    # 嵌入查表，支持任意前缀维度
    return weights[token_ids]


def silu(in_features: Tensor) -> Tensor:
    # SiLU = x * sigmoid(x)
    return in_features * torch.sigmoid(in_features)


def swiglu(in_features: Tensor, w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor) -> Tensor:
    # SwiGLU: silu(W1 x) * (W3 x) 再投影
    w1 = linear(in_features, w1_weight)
    w3 = linear(in_features, w3_weight)
    gated = silu(w1) * w3
    return linear(gated, w2_weight)


def softmax(in_features: Tensor, dim: int) -> Tensor:
    # 数值稳定的 softmax
    max_val = torch.amax(in_features, dim=dim, keepdim=True)
    exp = torch.exp(in_features - max_val)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    # 手写交叉熵，使用 log-sum-exp 稳定化
    max_val = torch.amax(inputs, dim=-1, keepdim=True)
    stabilized = inputs - max_val
    log_sum_exp = torch.log(torch.sum(torch.exp(stabilized), dim=-1, keepdim=True))
    log_probs = stabilized - log_sum_exp
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean()


def rmsnorm(in_features: Tensor, weights: Tensor, eps: float) -> Tensor:
    # RMSNorm：按最后一维做均方归一化，再应用缩放权重
    mean_sq = torch.mean(in_features * in_features, dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_sq + eps)
    return in_features * inv_rms * weights
