from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from yd_vector.data.svg_reader import read_svg_file
from yd_vector.data.svg_tokenizer.structured_tokens import BOS, EOS, PAD, SPECIAL_TOKENS, UNK
from yd_vector.data.svg_tokenizer.tokenizer_base import BaseTokenizer
from yd_vector.utils.io import iter_jsonl


class ByteTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        self.token_to_id = {PAD: 0, BOS: 1, EOS: 2, UNK: 3}

    @property
    def name(self) -> str:
        return "byte"

    @property
    def vocab_size(self) -> int:
        return 260

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True, max_length: int | None = None) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_token_id)
        ids.extend([b + 4 for b in text.encode("utf-8", errors="replace")])
        if add_eos:
            ids.append(self.eos_token_id)
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        buf = bytearray()
        for i in ids:
            if i < 4:
                if skip_special_tokens:
                    continue
                if i == self.unk_token_id:
                    buf.extend(b"?")
                continue
            if i <= 259:
                buf.append(i - 4)
            else:
                buf.extend(b"?")
        return buf.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "type": "byte",
                    "special_tokens": {PAD: 0, BOS: 1, EOS: 2, UNK: 3},
                    "vocab_size": self.vocab_size,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("type") != "byte":
            raise ValueError(f"Tokenizer at {path} is not byte")
        return cls()


def _replace_pair(tokens: List[str], pair: Tuple[str, str], merged: str) -> List[str]:
    out: List[str] = []
    i = 0
    a, b = pair
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


class SimpleBPETokenizer(BaseTokenizer):
    def __init__(self, merges: Sequence[Tuple[str, str]], token_to_id: Dict[str, int]) -> None:
        self.merges = list(merges)
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    @property
    def name(self) -> str:
        return "simple_bpe"

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @staticmethod
    def _byte_tokens(text: str) -> List[str]:
        return [f"b{b}" for b in text.encode("utf-8", errors="replace")]

    @staticmethod
    def _expand_token(token: str) -> List[int]:
        if token.startswith("b"):
            try:
                return [int(token[1:])]
            except ValueError:
                return [ord("?")]
        out: List[int] = []
        for part in token.split("|"):
            out.extend(SimpleBPETokenizer._expand_token(part))
        return out

    def _apply_merges(self, symbols: List[str]) -> List[str]:
        for a, b in self.merges:
            merged = f"{a}|{b}"
            symbols = _replace_pair(symbols, (a, b), merged)
        return symbols

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True, max_length: int | None = None) -> List[int]:
        symbols = self._apply_merges(self._byte_tokens(text))
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_token_id)
        ids.extend(self.token_to_id.get(s, self.unk_token_id) for s in symbols)
        if add_eos:
            ids.append(self.eos_token_id)
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        out = bytearray()
        for i in ids:
            if i in (self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id):
                if skip_special_tokens:
                    continue
                if i == self.unk_token_id:
                    out.extend(b"?")
                continue
            token = self.id_to_token.get(i, "")
            for b in self._expand_token(token):
                if 0 <= b <= 255:
                    out.append(b)
        return out.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "type": "simple_bpe",
                    "merges": self.merges,
                    "token_to_id": self.token_to_id,
                    "special_tokens": {PAD: 0, BOS: 1, EOS: 2, UNK: 3},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "SimpleBPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("type") != "simple_bpe":
            raise ValueError(f"Tokenizer at {path} is not simple_bpe")
        merges = [tuple(x) for x in data.get("merges", [])]
        return cls(merges=merges, token_to_id=data["token_to_id"])

    @classmethod
    def train_from_texts(cls, texts: Iterable[str], vocab_size: int = 1024, sample_size: int = 50000) -> "SimpleBPETokenizer":
        sequences: List[List[str]] = []
        seen = 0
        for text in texts:
            seq = cls._byte_tokens(text)
            if seq:
                sequences.append(seq)
                seen += 1
            if seen >= sample_size:
                break

        base_tokens = SPECIAL_TOKENS + [f"b{i}" for i in range(256)]
        target_merges = max(0, vocab_size - len(base_tokens))
        merges: List[Tuple[str, str]] = []

        for _ in range(target_merges):
            pair_counts: Counter[Tuple[str, str]] = Counter()
            for seq in sequences:
                if len(seq) < 2:
                    continue
                pair_counts.update(zip(seq[:-1], seq[1:]))
            if not pair_counts:
                break
            best_pair, freq = pair_counts.most_common(1)[0]
            if freq < 2:
                break
            merged = f"{best_pair[0]}|{best_pair[1]}"
            merges.append(best_pair)
            sequences = [_replace_pair(seq, best_pair, merged) for seq in sequences]

        token_list = base_tokens + [f"{a}|{b}" for a, b in merges]
        token_to_id = {tok: i for i, tok in enumerate(token_list)}
        return cls(merges=merges, token_to_id=token_to_id)


def train_bpe_from_manifest(
    manifest_path: str | Path,
    vocab_size: int = 1024,
    sample_size: int = 50000,
    max_file_bytes: int = 5 * 1024 * 1024,
    truncate_chars: int = 200000,
) -> SimpleBPETokenizer:
    def text_iter() -> Iterable[str]:
        seen = 0
        for row in iter_jsonl(manifest_path):
            path = row.get("path", "")
            if not path:
                continue
            r = read_svg_file(path, max_file_bytes=max_file_bytes, min_chars=1, truncate_chars=truncate_chars)
            if not r.ok:
                continue
            yield r.text
            seen += 1
            if seen >= sample_size:
                break

    return SimpleBPETokenizer.train_from_texts(text_iter(), vocab_size=vocab_size, sample_size=sample_size)
