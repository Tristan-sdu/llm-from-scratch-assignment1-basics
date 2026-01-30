from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import math
import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """全局梯度裁剪。

    方法：将所有参数梯度拼接成向量后按 L2 范数裁剪。
    关键变量：parameters 为参数集合；max_l2_norm 为阈值。
    解决问题：抑制梯度爆炸，提升训练稳定性。
    """
    # 复用 PyTorch 的全局范数裁剪
    # 将所有参数的梯度拼成一个向量后裁剪到 max_l2_norm
    torch.nn.utils.clip_grad_norm_(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """返回 AdamW 优化器类。

    方法：直接暴露 torch.optim.AdamW 类对象。
    关键变量：无。
    解决问题：便于外部按需实例化优化器。
    """
    # 直接返回 PyTorch AdamW 类，便于外部按需实例化
    return torch.optim.AdamW


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """线性 warmup + 余弦退火学习率。

    方法：warmup 阶段线性升到 max_lr，之后按余弦衰减到 min_lr。
    关键变量：it 为步数；warmup_iters/cosine_cycle_iters 控制阶段长度。
    解决问题：平滑训练初期并逐步降低学习率以收敛。
    """
    # 线性 warmup + 余弦退火
    # it 以 1 开始或 0 开始都可，但 warmup_iters=0 时应避免除零
    if it <= warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it >= cosine_cycle_iters:
        return min_learning_rate
    # 归一化到 [0, 1] 的进度，再做半周期余弦衰减
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine
