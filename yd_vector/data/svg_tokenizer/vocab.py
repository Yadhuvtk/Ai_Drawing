from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class Vocab:
    token_to_id: Dict[str, int] = field(default_factory=dict)

    @property
    def id_to_token(self) -> Dict[int, str]:
        return {v: k for k, v in self.token_to_id.items()}

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"token_to_id": self.token_to_id}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(token_to_id=data["token_to_id"])

    @classmethod
    def from_tokens(cls, tokens: List[str]) -> "Vocab":
        return cls(token_to_id={t: i for i, t in enumerate(tokens)})
