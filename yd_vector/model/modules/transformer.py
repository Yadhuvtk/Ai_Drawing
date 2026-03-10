from __future__ import annotations

import torch
import torch.nn as nn

from yd_vector.model.modules.attention import CausalSelfAttention
from yd_vector.model.modules.mlp import FeedForward
from yd_vector.model.modules.norms import build_norm


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_type: str = "rmsnorm",
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = build_norm(norm_type, d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, use_rope=use_rope)
        self.norm2 = build_norm(norm_type, d_model)
        self.mlp = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x
