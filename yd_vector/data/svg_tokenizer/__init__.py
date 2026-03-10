from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from yd_vector.data.svg_tokenizer.text_bpe import ByteTokenizer, SimpleBPETokenizer, train_bpe_from_manifest
from yd_vector.data.svg_tokenizer.tokenizer_base import BaseTokenizer


def load_tokenizer(path: str | Path) -> BaseTokenizer:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {p}")
    import json

    with open(p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    t = meta.get("type", "")
    if t == "byte":
        return ByteTokenizer.load(p)
    if t == "simple_bpe":
        return SimpleBPETokenizer.load(p)
    raise ValueError(f"Unsupported tokenizer type in {p}: {t}")


def build_tokenizer(cfg: Dict[str, Any], manifest_path: str | None = None) -> BaseTokenizer:
    tok_type = str(cfg.get("type", "byte")).lower()
    vocab_path = cfg.get("vocab_path")
    vocab_size = int(cfg.get("vocab_size", 260))

    if tok_type == "byte":
        tok = ByteTokenizer()
        if vocab_path:
            tok.save(vocab_path)
        return tok

    if tok_type == "simple_bpe":
        if vocab_path and Path(vocab_path).exists():
            return SimpleBPETokenizer.load(vocab_path)
        if manifest_path is None:
            raise ValueError("manifest_path is required to train simple_bpe tokenizer")
        tok = train_bpe_from_manifest(
            manifest_path=manifest_path,
            vocab_size=vocab_size,
            sample_size=int(cfg.get("bpe_sample_size", 50000)),
        )
        if vocab_path:
            tok.save(vocab_path)
        return tok

    raise ValueError(f"Unknown tokenizer type: {tok_type}")


__all__ = [
    "BaseTokenizer",
    "ByteTokenizer",
    "SimpleBPETokenizer",
    "train_bpe_from_manifest",
    "build_tokenizer",
    "load_tokenizer",
]
