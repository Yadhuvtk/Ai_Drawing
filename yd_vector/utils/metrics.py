from __future__ import annotations

import math
from typing import Iterable


def loss_to_perplexity(loss_value: float) -> float:
    if math.isnan(loss_value) or math.isinf(loss_value):
        return float("inf")
    return float(math.exp(min(20.0, loss_value)))


def svg_valid_ratio(texts: Iterable[str]) -> float:
    total = 0
    valid = 0
    for t in texts:
        total += 1
        lt = t.lower()
        if "<svg" in lt and "</svg>" in lt:
            valid += 1
    return (valid / total) if total > 0 else 0.0
