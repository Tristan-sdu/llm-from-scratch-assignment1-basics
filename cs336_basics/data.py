from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    if dataset.ndim != 1:
        raise ValueError("Dataset must be 1D.")
    # 随机采样起点，构造上下文与下一词标签
    max_start = len(dataset) - context_length
    start_indices = torch.randint(0, max_start, (batch_size,))
    offsets = torch.arange(context_length)
    x_idx = start_indices[:, None] + offsets[None, :]
    y_idx = x_idx + 1
    data = torch.from_numpy(dataset.astype(np.int64))
    x = data[x_idx]
    y = data[y_idx]
    return x.to(device), y.to(device)
