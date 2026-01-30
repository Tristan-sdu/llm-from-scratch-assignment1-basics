from __future__ import annotations

from typing import IO, BinaryIO
import os

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """保存训练检查点。

    方法：打包 model/optimizer state_dict 与迭代步数并用 torch.save 写入。
    关键变量：model/optimizer/iteration；out 为路径或文件对象。
    解决问题：支持训练中断后恢复。
    """
    # 同时保存模型、优化器和迭代步数
    # out 可以是路径或已打开的二进制文件对象
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    # torch.save 会根据文件后缀自动选择序列化格式
    torch.save(payload, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """加载训练检查点并恢复状态。

    方法：torch.load 读入 payload，恢复 state_dict，返回 iteration。
    关键变量：src 为路径或文件对象；model/optimizer 为待恢复对象。
    解决问题：继续训练或复现已有训练状态。
    """
    # 读取并恢复所有状态，返回已训练步数
    # 使用 map_location="cpu" 以保证跨设备加载安全
    payload = torch.load(src, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload["iteration"])
