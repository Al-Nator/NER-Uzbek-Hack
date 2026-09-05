"""Biaffine и GlobalPointer: оценки пар начального и конечного токенов."""

import math

import torch
from torch import nn
from torch.nn import functional as functional


class BiaffineHead(nn.Module):
    """Классифицирует каждую пару границ в NONE/ORG/NAME/GEO."""

    def __init__(self, hidden_size: int, head_size: int, classes: int) -> None:
        """Создаёт две MLP и билинейное преобразование с двумя bias."""
        super().__init__()
        self.start = nn.Sequential(nn.Linear(hidden_size, head_size), nn.GELU())
        self.end = nn.Sequential(nn.Linear(hidden_size, head_size), nn.GELU())
        self.weight = nn.Parameter(torch.empty(classes + 1, head_size + 1, head_size + 1))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Возвращает batch × classes × start × end, включая NONE."""
        start = functional.pad(self.start(hidden), (0, 1), value=1)
        end = functional.pad(self.end(hidden), (0, 1), value=1)
        return torch.einsum("bih,chd,bjd->bcij", start, self.weight, end)


def rotary_positions(value: torch.Tensor) -> torch.Tensor:
    """Применяет RoPE к batch × length × class × head_size."""
    length, size = value.shape[1], value.shape[-1]
    positions = torch.arange(length, device=value.device, dtype=torch.float32)
    frequencies = 10000 ** (-torch.arange(0, size, 2, device=value.device).float() / size)
    angles = positions[:, None] * frequencies[None, :]
    cosine = angles.cos()[None, :, None, :].to(value.dtype)
    sine = angles.sin()[None, :, None, :].to(value.dtype)
    even, odd = value[..., 0::2], value[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)


class GlobalPointerHead(nn.Module):
    """Полный GlobalPointer с отдельными Q/K каждого класса и RoPE."""

    def __init__(self, hidden_size: int, head_size: int, classes: int) -> None:
        """Создаёт проекцию пар векторов для каждого типа сущности."""
        super().__init__()
        self.classes, self.head_size = classes, head_size
        self.projection = nn.Linear(hidden_size, classes * head_size * 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Возвращает масштабированное скалярное произведение с RoPE."""
        projected = self.projection(hidden).reshape(*hidden.shape[:2], self.classes, -1)
        query, key = projected.chunk(2, dim=-1)
        return torch.einsum(
            "bich,bjch->bcij", rotary_positions(query), rotary_positions(key)
        ) / math.sqrt(self.head_size)


def global_pointer_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Считает multilabel categorical crossentropy в FP32 с маской -100."""
    scores, labels = logits.float().flatten(2), targets.flatten(2)
    zero = torch.zeros_like(scores[..., :1])
    negative = scores.masked_fill(labels != 0, -torch.inf)
    positive = (-scores).masked_fill(labels != 1, -torch.inf)
    return (
        torch.logsumexp(torch.cat((negative, zero), -1), -1)
        + torch.logsumexp(torch.cat((positive, zero), -1), -1)
    ).mean()
