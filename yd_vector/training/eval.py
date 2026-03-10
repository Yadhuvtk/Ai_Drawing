from __future__ import annotations

from typing import Dict

import torch

from yd_vector.utils.metrics import loss_to_perplexity, svg_valid_ratio


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    tokenizer,
    max_batches: int = 20,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()

    losses = []
    samples = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if loss is not None:
            losses.append(float(loss.item()))

        preds = torch.argmax(logits, dim=-1)
        for row in preds[:2]:
            samples.append(tokenizer.decode(row.tolist(), skip_special_tokens=True))

    mean_loss = float(sum(losses) / max(1, len(losses)))
    ppl = loss_to_perplexity(mean_loss)
    valid = svg_valid_ratio(samples)

    if was_training:
        model.train()

    return {"val_loss": mean_loss, "val_perplexity": ppl, "val_svg_valid_ratio": valid}
