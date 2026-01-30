from __future__ import annotations

import torch
from torch import Tensor


def linear(in_features: Tensor, weights: Tensor) -> Tensor:
    """线性投影（仿 PyTorch Linear 权重布局）。

    方法：按 (out, in) 权重转置后右乘输入。
    关键变量：in_features 为输入；weights 为权重矩阵。
    解决问题：实现最基础的仿射变换，供注意力/FFN 使用。
    """
    # 线性层：按 PyTorch Linear 的权重排布 (out, in)
    # in_features: (..., in), weights: (out, in) -> (..., out)
    return in_features @ weights.t()


def embedding(token_ids: Tensor, weights: Tensor) -> Tensor:
    """嵌入查表。

    方法：用 token_ids 作为索引从词表矩阵取向量。
    关键变量：token_ids 为索引；weights 为 (vocab, d_model)。
    解决问题：把离散 token 映射为连续向量表示。
    """
    # 嵌入查表，支持任意前缀维度
    # token_ids: (...), weights: (vocab, d_model) -> (..., d_model)
    return weights[token_ids]


def silu(in_features: Tensor) -> Tensor:
    """SiLU 激活函数。

    方法：x * sigmoid(x)。
    关键变量：in_features 为输入张量。
    解决问题：提供平滑非线性，提高 FFN 表达能力。
    """
    # SiLU = x * sigmoid(x)，常用于现代 Transformer FFN
    return in_features * torch.sigmoid(in_features)


def swiglu(in_features: Tensor, w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor) -> Tensor:
    """SwiGLU 前馈子层。

    方法：silu(W1 x) 与 (W3 x) 做逐元素乘法，再经 W2 投影。
    关键变量：w1/w3 控制门控分支；w2 将隐藏维映射回 d_model。
    解决问题：比传统 FFN 更强的门控表达。
    """
    # SwiGLU: silu(W1 x) * (W3 x) 再投影
    # 通常隐藏层维度为 d_ff，w2 将其映射回 d_model
    w1 = linear(in_features, w1_weight)
    w3 = linear(in_features, w3_weight)
    gated = silu(w1) * w3
    return linear(gated, w2_weight)


def softmax(in_features: Tensor, dim: int) -> Tensor:
    """数值稳定 softmax。

    方法：先减去最大值再 exp 归一化。
    关键变量：dim 指定归一化维度。
    解决问题：避免指数溢出，提高稳定性。
    """
    # 数值稳定的 softmax：减去最大值避免 exp 溢出
    max_val = torch.amax(in_features, dim=dim, keepdim=True)
    exp = torch.exp(in_features - max_val)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """手写交叉熵损失。

    方法：log-sum-exp 计算 log_probs，再取正确类别的负对数似然。
    关键变量：inputs 为 logits；targets 为目标类别索引。
    解决问题：评估分类/语言模型的预测误差。
    """
    # 手写交叉熵，使用 log-sum-exp 稳定化
    # inputs: (batch, vocab), targets: (batch,)
    max_val = torch.amax(inputs, dim=-1, keepdim=True)
    stabilized = inputs - max_val
    log_sum_exp = torch.log(torch.sum(torch.exp(stabilized), dim=-1, keepdim=True))
    log_probs = stabilized - log_sum_exp
    # 仅取正确类别的负对数似然
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean()


def rmsnorm(in_features: Tensor, weights: Tensor, eps: float) -> Tensor:
    """RMSNorm 归一化层。

    方法：按最后一维计算均方根并缩放，再乘以权重。
    关键变量：weights 为缩放参数；eps 防止除零。
    解决问题：稳定训练并保持激活尺度。
    """
    # RMSNorm：按最后一维做均方归一化，再应用缩放权重
    # 与 LayerNorm 不同：不减均值，只按均方根缩放
    mean_sq = torch.mean(in_features * in_features, dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_sq + eps)
    return in_features * inv_rms * weights
