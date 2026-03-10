from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from yd_vector.data.svg_reader import read_svg_file
from yd_vector.data.svg_tokenizer.tokenizer_base import BaseTokenizer


class IM2SVGDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        splits_path: str | Path,
        tokenizer: BaseTokenizer,
        split: str = "train",
        image_size: int = 256,
        max_seq_len: int = 1024,
        max_file_bytes: int = 5 * 1024 * 1024,
        min_chars: int = 20,
        truncate_chars: int = 200000,
        max_records: Optional[int] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.splits_path = Path(splits_path)
        self.tokenizer = tokenizer
        self.split = split
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.max_file_bytes = max_file_bytes
        self.min_chars = min_chars
        self.truncate_chars = truncate_chars
        self.max_records = max_records

        with open(self.splits_path, "r", encoding="utf-8") as f:
            split_obj = json.load(f)

        allowed_ids = set(split_obj[split])

        self.rows: List[Dict] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("id") in allowed_ids:
                    self.rows.append(row)

        if self.max_records is not None:
            self.rows = self.rows[: self.max_records]

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        tensor = torch.from_numpy(arr)
        return tensor

    def _load_svg_tokens(self, svg_path: str) -> np.ndarray:
        r = read_svg_file(
            svg_path,
            max_file_bytes=self.max_file_bytes,
            min_chars=self.min_chars,
            truncate_chars=self.truncate_chars,
        )
        if not r.ok:
            raise ValueError(f"Failed to read SVG: {svg_path}")

        ids = self.tokenizer.encode(r.text, add_bos=True, add_eos=True)
        arr = np.asarray(ids, dtype=np.int64)

        if len(arr) > self.max_seq_len + 1:
            if self.split == "train":
                st = random.randint(0, len(arr) - (self.max_seq_len + 1))
                arr = arr[st : st + self.max_seq_len + 1]
            else:
                arr = arr[: self.max_seq_len + 1]

        return arr

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]

        pixel_values = self._load_image(row["png_path"])
        ids = self._load_svg_tokens(row["svg_path"])

        x = torch.from_numpy(ids[:-1].copy()).long()
        y = torch.from_numpy(ids[1:].copy()).long()
        mask = torch.ones_like(x, dtype=torch.long)

        return {
            "id": row["id"],
            "pixel_values": pixel_values,
            "input_ids": x,
            "attention_mask": mask,
            "labels": y,
        }