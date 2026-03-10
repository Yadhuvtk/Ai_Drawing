from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator

from yd_vector.utils.io import iter_jsonl, write_json, write_jsonl


@dataclass
class ManifestRecord:
    path: str
    size_bytes: int
    chars: int
    split: str
    has_svg_tags: bool = True
    normalized: bool = False
    error: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def save_manifest(records: Iterable[ManifestRecord], path: str | Path) -> None:
    write_jsonl((r.to_dict() for r in records), path)


def iter_manifest(path: str | Path) -> Iterator[Dict]:
    yield from iter_jsonl(path)


def save_splits(data: Dict, path: str | Path) -> None:
    write_json(data, path, indent=2)
