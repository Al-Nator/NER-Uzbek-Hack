"""Три лёгких варианта отбора spans без дополнительного Transformer encoder-а."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RerankerConfig:
    """Фиксированный бюджет и варианты признаков пилотного reranker."""

    run_id: str
    variant: str
    seed: int = 42
    epochs: int = 12
    batch_size: int = 512
    learning_rate: float = 0.001
    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)

    def __post_init__(self) -> None:
        """Проверяет поддерживаемые варианты и численные параметры."""
        if self.variant not in {"linear", "mlp", "char_context"}:
            raise ValueError("Неизвестный вариант reranker")
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("Некорректный бюджет reranker")
        if not self.thresholds or any(not 0 < t < 1 for t in self.thresholds):
            raise ValueError("Некорректные пороги reranker")


class SpanReranker(nn.Module):
    """Оценивает правильность класса и точных границ каждого кандидата."""

    def __init__(self, features: int, variant: str):
        """Создаёт линейную, MLP или MLP+char-CNN модель."""
        super().__init__()
        self.variant = variant
        if variant == "char_context":
            self.embedding = nn.Embedding(4096, 24, padding_idx=0)
            self.convolution = nn.Conv1d(24, 48, 3, padding=1)
            features += 48
        self.head = (
            nn.Linear(features, 1)
            if variant == "linear"
            else nn.Sequential(
                nn.Linear(features, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
            )
        )

    def forward(self, features: torch.Tensor, characters: torch.Tensor) -> torch.Tensor:
        """Возвращает один logit правильности для каждого exact-span кандидата."""
        if self.variant == "char_context":
            hidden = torch.nn.functional.gelu(
                self.convolution(self.embedding(characters).transpose(1, 2))
            )
            hidden = hidden.masked_fill((characters == 0)[:, None], -1e4).amax(-1)
            features = torch.cat((features, hidden), dim=1)
        return self.head(features).squeeze(-1)
