from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from yd_vector.inference.generate import generate_tokens
from yd_vector.model.config import ModelConfig
from yd_vector.model.modules.norms import build_norm
from yd_vector.model.modules.transformer import DecoderBlock
from yd_vector.model.vision.adapter import VisionAdapter
from yd_vector.model.vision.vit import TinyViT


class YDVectorForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.use_vision = config.use_vision
        self.total_max_seq_len = config.max_seq_len + (config.vision_prefix_tokens if config.use_vision else 0)

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = None if config.rope else nn.Embedding(self.total_max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.vision_encoder = TinyViT(in_channels=3, d_model=config.d_model) if config.use_vision else None
        self.vision_adapter = (
            VisionAdapter(d_model=config.d_model, prefix_tokens=config.vision_prefix_tokens) if config.use_vision else None
        )

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    norm_type=config.norm_type,
                    use_rope=config.rope,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm_f = build_norm(config.norm_type, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def _build_input_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
        bsz, seq_len = input_ids.shape
        x = self.token_embedding(input_ids)
        prefix_len = 0

        if attention_mask is None:
            attention_mask = torch.ones((bsz, seq_len), dtype=torch.long, device=input_ids.device)

        if self.use_vision and pixel_values is not None:
            assert self.vision_encoder is not None
            assert self.vision_adapter is not None
            vision_features = self.vision_encoder(pixel_values)
            vision_prefix = self.vision_adapter(vision_features).to(dtype=x.dtype)
            prefix_len = vision_prefix.shape[1]
            x = torch.cat([vision_prefix, x], dim=1)

            prefix_mask = torch.ones((bsz, prefix_len), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            if labels is not None:
                prefix_labels = torch.full((bsz, prefix_len), -100, dtype=labels.dtype, device=labels.device)
                labels = torch.cat([prefix_labels, labels], dim=1)

        if self.pos_embedding is not None:
            total_seq_len = x.shape[1]
            pos = torch.arange(total_seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
            x = x + self.pos_embedding(pos)

        x = self.drop(x)
        return x, attention_mask, labels, prefix_len

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        x, attention_mask, labels, prefix_len = self._build_input_sequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
        )

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.norm_f(x)
        logits = self.lm_head(x)
        output_logits = logits[:, prefix_len:, :] if prefix_len > 0 else logits

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return loss, output_logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int = 2,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return generate_tokens(
            model=self,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            pixel_values=pixel_values,
        )
