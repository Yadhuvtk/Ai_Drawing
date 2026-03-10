from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class BaseTokenizer(ABC):
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True, max_length: int | None = None) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError
