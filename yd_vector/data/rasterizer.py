from __future__ import annotations

from pathlib import Path


def has_cairosvg() -> bool:
    try:
        import cairosvg  # noqa: F401

        return True
    except Exception:
        return False


def rasterize_svg_to_png(svg_text: str, out_path: str | Path, width: int = 512, height: int = 512) -> bool:
    try:
        import cairosvg
    except Exception:
        return False
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(bytestring=svg_text.encode("utf-8", errors="replace"), write_to=str(p), output_width=width, output_height=height)
    return True
