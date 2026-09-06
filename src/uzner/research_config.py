"""Изолированные настройки проверяемых span-абляций."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SpanResearchConfig:
    """Вес каждого дополнительного сигнала; нули сохраняют reference."""

    bioes_crf_weight: float = 0.0
    boundary_weight: float = 0.0
    smoothing: float = 0.0
    hard_negative_weight: float = 0.0
    hard_negative_topk: int = 32

    def __post_init__(self) -> None:
        """Запрещает отрицательные веса и некорректное сглаживание."""
        if min(self.bioes_crf_weight, self.boundary_weight, self.hard_negative_weight) < 0:
            raise ValueError("Веса дополнительных losses должны быть неотрицательны")
        if not 0 <= self.smoothing < 0.5 or self.hard_negative_topk < 1:
            raise ValueError("Некорректные smoothing или hard_negative_topk")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SpanResearchConfig":
        """Создаёт типизированные настройки с запретом неизвестных ключей."""
        return cls(**value)

    @property
    def enabled(self) -> bool:
        """Проверяет наличие ненулевой абляции."""
        return any(
            (self.bioes_crf_weight, self.boundary_weight, self.smoothing, self.hard_negative_weight)
        )
