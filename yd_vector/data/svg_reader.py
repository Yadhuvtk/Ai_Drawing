from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SVGReadResult:
    ok: bool
    text: str = ""
    size_bytes: int = 0
    encoding: str = ""
    error: str = ""


def _decode_bytes(raw: bytes) -> tuple[str, str]:
    for enc in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace"), "latin-1"


def read_svg_file(
    path: str | Path,
    max_file_bytes: int = 5 * 1024 * 1024,
    min_chars: int = 20,
    truncate_chars: int | None = None,
) -> SVGReadResult:
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError as e:
        return SVGReadResult(ok=False, error=f"stat_error:{e}")

    if size > max_file_bytes:
        return SVGReadResult(ok=False, size_bytes=size, error="file_too_large")

    try:
        raw = p.read_bytes()
    except OSError as e:
        return SVGReadResult(ok=False, size_bytes=size, error=f"read_error:{e}")

    text, encoding = _decode_bytes(raw)
    text = text.replace("\x00", "")
    if truncate_chars is not None and len(text) > truncate_chars:
        text = text[:truncate_chars]

    if len(text) < min_chars:
        return SVGReadResult(ok=False, size_bytes=size, encoding=encoding, error="too_short")

    return SVGReadResult(ok=True, text=text, size_bytes=size, encoding=encoding)
