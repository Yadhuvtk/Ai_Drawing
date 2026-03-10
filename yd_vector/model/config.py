from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ModelConfig:
    vocab_size: int = 260
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    max_seq_len: int = 1024
    rope: bool = True
    tie_embeddings: bool = True
    norm_type: str = "rmsnorm"

    # new for image -> svg
    use_vision: bool = False
    image_size: int = 256
    vision_prefix_tokens: int = 4

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        fields = cls.__dataclass_fields__.keys()
        return cls(**{k: data[k] for k in fields if k in data})