from __future__ import annotations

from contextlib import nullcontext

import torch


def build_scaler(enabled: bool) -> torch.cuda.amp.GradScaler:
    return torch.cuda.amp.GradScaler(enabled=enabled and torch.cuda.is_available())


def autocast_context(enabled: bool):
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()
