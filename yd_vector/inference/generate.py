from __future__ import annotations

import torch

from yd_vector.inference.constraints import clamp_temperature, clamp_top_p


def _sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    cutoff = cumulative > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_tokens(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.9,
    eos_token_id: int = 2,
    pixel_values: torch.Tensor | None = None,
) -> torch.Tensor:
    was_training = model.training
    model.eval()

    temperature = clamp_temperature(temperature)
    top_p = clamp_top_p(top_p)
    out = input_ids

    for _ in range(max_new_tokens):
        ctx = out
        if ctx.size(1) > model.config.max_seq_len:
            ctx = ctx[:, -model.config.max_seq_len :]
        attention_mask = torch.ones_like(ctx, dtype=torch.long, device=ctx.device)
        _, logits = model(ctx, attention_mask=attention_mask, labels=None, pixel_values=pixel_values)
        next_logits = logits[:, -1, :]

        greedy = temperature == 0.0 or top_p == 0.0
        if greedy:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            scaled = next_logits / max(1e-6, temperature)
            if top_p >= 1.0:
                probs = torch.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = _sample_top_p(scaled, top_p=top_p)

        out = torch.cat([out, next_token], dim=1)
        if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
            break

    if was_training:
        model.train()
    return out
