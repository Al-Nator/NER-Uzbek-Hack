"""Маленькие тестовые tokenizer и encoder без сети."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from uzner.domain import Document, Entity


class FakeTokenizer:
    """Имитирует fast tokenizer с word offsets и sliding windows."""

    pad_token_id = 0
    is_fast = True

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        stride: int = 0,
        return_offsets_mapping: bool = False,
        return_overflowing_tokens: bool = False,
        padding: bool = False,
    ) -> dict[str, Any]:
        """Токенизирует непустые фрагменты и возвращает offsets."""
        del padding
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        ids = [index + 10 for index in range(len(offsets))]
        if not truncation:
            return {
                "input_ids": ids,
                "offset_mapping": offsets if return_offsets_mapping else None,
            }
        if max_length is None:
            raise ValueError("max_length обязателен при truncation")
        capacity = max_length - (2 if add_special_tokens else 0)
        step = capacity - stride
        if capacity < 1 or step < 1:
            raise ValueError("Некорректное окно FakeTokenizer")
        chunks: list[tuple[list[int], list[tuple[int, int]]]] = []
        start = 0
        while start < len(ids):
            chunk_ids = ids[start : start + capacity]
            chunk_offsets = offsets[start : start + capacity]
            if add_special_tokens:
                chunk_ids = [1, *chunk_ids, 2]
                chunk_offsets = [(0, 0), *chunk_offsets, (0, 0)]
            chunks.append((chunk_ids, chunk_offsets))
            if start + capacity >= len(ids):
                break
            start += step
        encoded = {
            "input_ids": [chunk[0] for chunk in chunks],
            "attention_mask": [[1] * len(chunk[0]) for chunk in chunks],
            "token_type_ids": [[0] * len(chunk[0]) for chunk in chunks],
            "offset_mapping": [chunk[1] for chunk in chunks],
        }
        if return_overflowing_tokens:
            return encoded
        return {key: value[0] for key, value in encoded.items()}

    def save_pretrained(self, path: Path) -> None:
        """Создаёт маркер тестового tokenizer-а."""
        path.mkdir(parents=True, exist_ok=True)
        (path / "fake_tokenizer.json").write_text("{}", encoding="utf-8")


class TinyEncoder(nn.Module):
    """Маленький encoder с Transformers-подобным выходом."""

    def __init__(self, hidden_size: int = 8, max_positions: int = 64) -> None:
        """Создаёт embedding и минимальный config."""
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            max_position_embeddings=max_positions,
        )
        self.embedding = nn.Embedding(128, hidden_size)
        self.checkpointing_enabled = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        """Возвращает embedding как last_hidden_state."""
        del attention_mask, token_type_ids
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    def gradient_checkpointing_enable(self) -> None:
        """Отмечает включение gradient checkpointing."""
        self.checkpointing_enabled = True


def sample_documents() -> tuple[Document, ...]:
    """Возвращает два коротких размеченных документа."""
    return (
        Document(
            hash="d1",
            text="Ali Toshkentga bordi",
            entities=(
                Entity(label="NAME", start=0, end=3),
                Entity(label="GEO", start=4, end=14),
            ),
            source="test",
        ),
        Document(
            hash="d2",
            text="ACME Samarqand",
            entities=(
                Entity(label="ORG", start=0, end=4),
                Entity(label="GEO", start=5, end=14),
            ),
            source="test",
        ),
    )
