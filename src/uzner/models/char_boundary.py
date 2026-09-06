"""Небольшая символьная голова уточнения границ вокруг frozen token anchors."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class CharConfig:
    """Параметры символьной абляции с фиксированным радиусом в Unicode-символах."""

    radius: int = 4
    context: int = 6
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    max_examples: int = 100_000
    confidence: float = 0.6
    seed: int = 42

    def __post_init__(self) -> None:
        """Запрещает пустые окна и невозможную уверенность."""
        if min(self.radius, self.context, self.epochs, self.batch_size, self.max_examples) < 1:
            raise ValueError("Параметры размеров должны быть положительными")
        if not 0 <= self.confidence <= 1 or self.learning_rate <= 0:
            raise ValueError("Некорректный confidence или LR")


def character_window(
    text: str, anchor: int, alphabet: dict[str, int], config: CharConfig
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Читает исходные Unicode-символы, не нормализуя текст или координаты."""
    reach = config.radius + config.context
    chars = tuple(
        alphabet.get(text[p], 1) if 0 <= p < len(text) else 0
        for p in range(anchor - reach, anchor + reach)
    )
    valid = tuple(
        0 <= anchor + delta <= len(text) for delta in range(-config.radius, config.radius + 1)
    )
    return chars, valid


class CharBoundaryHead(nn.Module):
    """Контекстный char-CNN оценивает каждую границу между соседними символами."""

    def __init__(self, hidden_size: int, alphabet_size: int, config: CharConfig) -> None:
        """Создаёт лёгкую голову, не содержащую token encoder."""
        super().__init__()
        self.config = config
        self.characters = nn.Embedding(alphabet_size, 32, padding_idx=0)
        self.conv = nn.Conv1d(32, 32, 3, padding=1)
        self.token = nn.Linear(hidden_size, 64)
        self.kind = nn.Embedding(6, 16)
        self.position = nn.Embedding(2 * config.radius + 1, 8)
        self.scorer = nn.Sequential(nn.Linear(152, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(
        self, hidden: torch.Tensor, chars: torch.Tensor, kind: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Оценивает точные символьные offsets, исключая позиции за текстом."""
        states = functional.gelu(self.conv(self.characters(chars).transpose(1, 2)).transpose(1, 2))
        width, left = 2 * self.config.radius + 1, self.config.context
        character_pair = torch.cat(
            (states[:, left - 1 : left - 1 + width], states[:, left : left + width]), -1
        )
        token = self.token(hidden.float())[:, None].expand(-1, width, -1)
        kinds = self.kind(kind)[:, None].expand(-1, width, -1)
        positions = self.position(torch.arange(width, device=chars.device))[None].expand(
            len(chars), -1, -1
        )
        scores = self.scorer(torch.cat((character_pair, token, kinds, positions), -1)).squeeze(-1)
        return scores.masked_fill(~valid, -1e4)
