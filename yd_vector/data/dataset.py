from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from yd_vector.data.svg_reader import read_svg_file
from yd_vector.data.svg_tokenizer.tokenizer_base import BaseTokenizer


def count_split_records(manifest_path: str | Path, split: str) -> int:
    n = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("split") == split:
                n += 1
    return n


class SVGIterableDataset(IterableDataset):
    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: BaseTokenizer,
        split: str = "train",
        max_seq_len: int = 1024,
        max_file_bytes: int = 5 * 1024 * 1024,
        min_chars: int = 20,
        truncate_chars: int = 200000,
        cache_dir: str | Path | None = None,
        cache_enabled: bool = False,
        repeat: bool = False,
        max_records: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
        self.split = split
        self.max_seq_len = max_seq_len
        self.max_file_bytes = max_file_bytes
        self.min_chars = min_chars
        self.truncate_chars = truncate_chars
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_enabled = cache_enabled and self.cache_dir is not None
        self.repeat = repeat
        self.max_records = max_records
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, file_path: str, file_size: int) -> Path | None:
        if not self.cache_enabled or self.cache_dir is None:
            return None
        key = f"{file_path}|{file_size}|{self.tokenizer.name}|{self.max_seq_len}|{self.truncate_chars}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def _read_or_tokenize(self, file_path: str, file_size: int = 0) -> np.ndarray | None:
        cpath = self._cache_path(file_path, file_size=file_size)
        if cpath is not None and cpath.exists():
            try:
                arr = np.load(cpath, allow_pickle=False)
                return arr.astype(np.int64, copy=False)
            except Exception:
                pass

        r = read_svg_file(
            file_path,
            max_file_bytes=self.max_file_bytes,
            min_chars=self.min_chars,
            truncate_chars=self.truncate_chars,
        )
        if not r.ok:
            return None
        ids = self.tokenizer.encode(r.text, add_bos=True, add_eos=True)
        arr = np.asarray(ids, dtype=np.int64)
        if cpath is not None:
            try:
                np.save(cpath, arr)
            except Exception:
                pass
        return arr

    def _iter_rows(self) -> Iterator[Dict]:
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row["_line_no"] = i
                yield row

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker = get_worker_info()
        wid = worker.id if worker is not None else 0
        nworkers = worker.num_workers if worker is not None else 1

        yielded = 0
        while True:
            for row in self._iter_rows():
                if row["_line_no"] % nworkers != wid:
                    continue
                if row.get("split") != self.split:
                    continue
                path = row.get("path")
                if not path:
                    continue
                fsize = int(row.get("size_bytes", 0))
                ids = self._read_or_tokenize(path, file_size=fsize)
                if ids is None or len(ids) < 2:
                    continue
                if len(ids) > self.max_seq_len + 1:
                    if self.split == "train":
                        st = random.randint(0, len(ids) - (self.max_seq_len + 1))
                        ids = ids[st : st + self.max_seq_len + 1]
                    else:
                        ids = ids[: self.max_seq_len + 1]

                x = torch.from_numpy(ids[:-1].copy()).long()
                y = torch.from_numpy(ids[1:].copy()).long()
                mask = torch.ones_like(x, dtype=torch.long)
                yield {"input_ids": x, "attention_mask": mask, "labels": y}
                yielded += 1
                if self.max_records is not None and yielded >= self.max_records:
                    return
            if not self.repeat:
                return
