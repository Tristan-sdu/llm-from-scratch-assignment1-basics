from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import math
import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    # 复用 PyTorch 的全局范数裁剪
    torch.nn.utils.clip_grad_norm_(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    # 直接返回 PyTorch AdamW 类
    return torch.optim.AdamW


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    # 线性 warmup + 余弦退火
    if it <= warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it >= cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine
