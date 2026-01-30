from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    """从一维 token 序列中采样训练 batch。

    方法：随机采样起点，切出长度为 context_length 的片段作为 x，右移一位作为 y。
    关键变量：dataset 为一维 token 数组；batch_size/ context_length 控制批量与上下文长度。
    解决问题：构造自回归语言模型的输入与下一词标签。
    """
    if dataset.ndim != 1:
        raise ValueError("Dataset must be 1D.")
    # 随机采样起点，构造上下文与下一词标签
    # dataset 是一维 token 序列；x 为长度 context_length 的上下文，y 为对应的下一个 token
    max_start = len(dataset) - context_length
    # 起点索引：每条样本独立抽样
    start_indices = torch.randint(0, max_start, (batch_size,))
    # 偏移 [0..context_length-1]，用于拼出连续片段
    offsets = torch.arange(context_length)
    x_idx = start_indices[:, None] + offsets[None, :]
    # 标签为右移 1 位
    y_idx = x_idx + 1
    # numpy -> torch，并确保是 int64 以作为索引/分类标签
    data = torch.from_numpy(dataset.astype(np.int64))
    x = data[x_idx]
    y = data[y_idx]
    # 返回搬到目标设备上的 batch
    return x.to(device), y.to(device)
