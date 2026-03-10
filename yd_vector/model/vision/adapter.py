from __future__ import annotations

import torch
import torch.nn as nn


class VisionAdapter(nn.Module):
    def __init__(self, d_model: int = 384, prefix_tokens: int = 4) -> None:
        super().__init__()
        self.prefix_tokens = prefix_tokens
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * prefix_tokens),
            nn.GELU(),
            nn.Linear(d_model * prefix_tokens, d_model * prefix_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d_model]
        x = self.proj(x)
        x = x.view(x.shape[0], self.prefix_tokens, -1)  # [B, P, d_model]
        return x