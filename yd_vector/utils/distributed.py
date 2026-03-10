from __future__ import annotations

import torch


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def world_size() -> int:
    return torch.distributed.get_world_size() if is_distributed() else 1


def rank() -> int:
    return torch.distributed.get_rank() if is_distributed() else 0
