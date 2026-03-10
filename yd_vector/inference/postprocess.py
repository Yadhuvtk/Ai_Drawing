from __future__ import annotations

import re


SVG_BLOCK_RE = re.compile(r"<svg\b.*?</svg>", re.IGNORECASE | re.DOTALL)
SVG_OPEN_RE = re.compile(r"<svg\b([^>]*)>", re.IGNORECASE | re.DOTALL)

# Capture d=... even if it's broken/unclosed: stop at quote or </svg> or >
PATH_D_SINGLE_RE = re.compile(r"<path\b[^>]*\bd\s*=\s*'([^']*)", re.IGNORECASE | re.DOTALL)
PATH_D_DOUBLE_RE = re.compile(r'<path\b[^>]*\bd\s*=\s*"([^"]*)', re.IGNORECASE | re.DOTALL)


def _extract_first_svg_block(text: str) -> str:
    m = SVG_BLOCK_RE.search(text)
    return m.group(0) if m else text


def _sanitize_attr_value(s: str) -> str:
    # Remove characters that break XML inside attribute values
    return s.replace("<", " ").replace(">", " ").replace("&", " ")


def _extract_svg_open_attrs(text: str) -> str | None:
    m = SVG_OPEN_RE.search(text)
    if not m:
        return None
    attrs = m.group(1).strip()
    # sanitize any stray angle brackets in attrs
    attrs = _sanitize_attr_value(attrs)
    return attrs


def _extract_path_d(text: str) -> str | None:
    m = PATH_D_SINGLE_RE.search(text)
    if m:
        return m.group(1)
    m = PATH_D_DOUBLE_RE.search(text)
    if m:
        return m.group(1)
    return None


def _rebuild_minimal_svg(svg_open_attrs: str | None, d: str | None) -> str:
    # Provide a safe default SVG wrapper if none found
    if not svg_open_attrs:
        svg_open_attrs = "xmlns='http://www.w3.org/2000/svg' viewBox='0 0 256 256'"

    # If we couldn't extract d, just create a tiny visible mark (so it renders)
    if not d:
        d = "M 10 10 L 246 10"

    d = _sanitize_attr_value(d)

    # Force visibility: add stroke; many generated paths are lines / unclosed shapes
    return (
        f"<svg {svg_open_attrs}>"
        f"<path d='{d}' fill='none' stroke='black' stroke-width='4' stroke-linecap='round' stroke-linejoin='round'/>"
        f"</svg>"
    )


def ensure_svg_wrapped(
    text: str,
    fallback_prefix: str = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 256 256'>",
    fallback_suffix: str = "</svg>",
) -> str:
    """
    Robust postprocess:
    - Try to keep first SVG block
    - Extract svg attrs + first path d
    - Rebuild a minimal safe SVG (prevents XML parse errors forever)
    """
    text = _extract_first_svg_block(text)

    svg_attrs = _extract_svg_open_attrs(text)
    d = _extract_path_d(text)

    # If model didn't produce <svg>, prepend fallback so attrs exist
    if svg_attrs is None and "<svg" not in text.lower():
        text = fallback_prefix + text + fallback_suffix
        svg_attrs = _extract_svg_open_attrs(text)

    return _rebuild_minimal_svg(svg_attrs, d)