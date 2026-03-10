from __future__ import annotations

from typing import Dict, List

import torch


def causal_lm_collate(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    if not batch:
        return {
            "input_ids": torch.empty(0, dtype=torch.long),
            "attention_mask": torch.empty(0, dtype=torch.long),
            "labels": torch.empty(0, dtype=torch.long),
        }

    max_len = max(x["input_ids"].shape[0] for x in batch)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), fill_value=pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), fill_value=-100, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["input_ids"].shape[0]
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
        attention_mask[i, :n] = 1

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
