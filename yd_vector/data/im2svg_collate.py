from __future__ import annotations

from typing import Dict, List

import torch


def im2svg_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)

    max_len = max(x["input_ids"].shape[0] for x in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for b in batch:
        x = b["input_ids"]
        m = b["attention_mask"]
        y = b["labels"]

        pad_len = max_len - x.shape[0]

        if pad_len > 0:
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)], dim=0)
            m = torch.cat([m, torch.zeros(pad_len, dtype=torch.long)], dim=0)
            y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)], dim=0)

        input_ids.append(x)
        attention_mask.append(m)
        labels.append(y)

    return {
        "pixel_values": pixel_values,
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "labels": torch.stack(labels, dim=0),
    }