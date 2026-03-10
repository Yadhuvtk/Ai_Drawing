from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from yd_vector.model.modules.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_rope: bool = True) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim) if use_rope else None

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope.apply_rotary(q, k)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        causal = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~causal[None, None, :, :], float("-inf"))

        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(torch.bool)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(dtype=q.dtype)
        attn_probs = self.attn_dropout(attn_probs)
        out = attn_probs @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out
