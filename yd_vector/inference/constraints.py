from __future__ import annotations


def clamp_top_p(top_p: float) -> float:
    return max(0.0, min(1.0, top_p))


def clamp_temperature(temp: float) -> float:
    return max(0.0, float(temp))
