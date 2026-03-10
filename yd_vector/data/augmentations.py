from __future__ import annotations


def maybe_augment_svg_text(text: str, enabled: bool = False) -> str:
    return text if enabled else text
