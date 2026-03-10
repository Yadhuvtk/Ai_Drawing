from __future__ import annotations

import re
from dataclasses import dataclass

XML_DECL_RE = re.compile(r"<\?xml.*?\?>", flags=re.IGNORECASE | re.DOTALL)
DOCTYPE_RE = re.compile(r"<!DOCTYPE.*?>", flags=re.IGNORECASE | re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
METADATA_RE = re.compile(r"<metadata[^>]*>.*?</metadata>", flags=re.IGNORECASE | re.DOTALL)
SVG_BLOCK_RE = re.compile(r"(<svg\b[^>]*>.*?</svg>)", flags=re.IGNORECASE | re.DOTALL)
FLOAT_RE = re.compile(r"(?<![\w.-])[-+]?\d*\.\d+([eE][-+]?\d+)?")


@dataclass
class NormalizeResult:
    ok: bool
    text: str = ""
    error: str = ""


def _round_float_match(match: re.Match[str], precision: int) -> str:
    s = match.group(0)
    try:
        v = float(s)
    except ValueError:
        return s
    out = f"{v:.{precision}f}".rstrip("0").rstrip(".")
    if out == "-0":
        out = "0"
    return out


def normalize_svg_text(text: str, float_precision: int | None = None) -> NormalizeResult:
    if not text:
        return NormalizeResult(ok=False, error="empty_text")

    cleaned = text.replace("\ufeff", "")
    cleaned = XML_DECL_RE.sub("", cleaned)
    cleaned = DOCTYPE_RE.sub("", cleaned)
    cleaned = COMMENT_RE.sub("", cleaned)
    cleaned = METADATA_RE.sub("", cleaned)

    m = SVG_BLOCK_RE.search(cleaned)
    if not m:
        return NormalizeResult(ok=False, error="no_svg_block")
    svg = m.group(1)
    svg = re.sub(r">\s+<", "><", svg)
    svg = re.sub(r"\s{2,}", " ", svg).strip()

    if float_precision is not None and float_precision >= 0:
        svg = FLOAT_RE.sub(lambda x: _round_float_match(x, float_precision), svg)

    lower = svg.lower()
    if "<svg" not in lower or "</svg>" not in lower:
        return NormalizeResult(ok=False, error="invalid_svg_tags")
    return NormalizeResult(ok=True, text=svg)
