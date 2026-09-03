"""Строгие доменные типы exact-span NER."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

LABELS = ("ORG", "NAME", "GEO")
LABEL_SET = frozenset(LABELS)


@dataclass(frozen=True, slots=True)
class Entity:
    """Одна сущность в символьных координатах исходного текста."""

    start: int
    end: int
    label: str
    score: float | None = None

    def __post_init__(self) -> None:
        """Проверяет класс, границы и необязательную уверенность."""
        if self.label not in LABEL_SET:
            raise ValueError(f"Неизвестный класс сущности: {self.label!r}")
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("start должен быть целым числом")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise TypeError("end должен быть целым числом")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Ожидается непустой интервал 0 <= start < end")
        if self.score is not None and not isfinite(self.score):
            raise ValueError("score должен быть конечным числом")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Entity:
        """Создаёт сущность из JSON-подобного отображения."""
        return cls(
            label=value.get("label"),
            start=value.get("start"),
            end=value.get("end"),
            score=value.get("score"),
        )

    def to_mapping(self, *, include_score: bool = False) -> dict[str, Any]:
        """Возвращает сущность в формате API и JSONL."""
        result: dict[str, Any] = {
            "label": self.label,
            "start": self.start,
            "end": self.end,
        }
        if include_score and self.score is not None:
            result["score"] = self.score
        return result


@dataclass(frozen=True, slots=True)
class Document:
    """Один размеченный документ с указанием источника данных."""

    hash: str
    text: str
    entities: tuple[Entity, ...] = ()
    source: str = "unknown"

    def __post_init__(self) -> None:
        """Нормализует порядок и проверяет плоскую gold-разметку."""
        if not isinstance(self.hash, str) or not self.hash:
            raise ValueError("hash документа должен быть непустой строкой")
        if not isinstance(self.text, str):
            raise TypeError("text документа должен быть строкой")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source документа должен быть непустой строкой")

        ordered = tuple(sorted(self.entities, key=lambda item: (item.start, item.end, item.label)))
        object.__setattr__(self, "entities", ordered)
        previous: Entity | None = None
        for entity in ordered:
            if entity.end > len(self.text):
                raise ValueError(f"Сущность {entity!r} выходит за длину текста {len(self.text)}")
            if previous is not None and entity.start < previous.end:
                raise ValueError(f"Сущности пересекаются: {previous!r} и {entity!r}")
            previous = entity

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, source: str) -> Document:
        """Создаёт документ из записи исходного JSONL."""
        raw_entities = value.get("entities", [])
        if not isinstance(raw_entities, list):
            raise TypeError("entities должен быть массивом")
        return cls(
            hash=value.get("hash"),
            text=value.get("text"),
            entities=tuple(Entity.from_mapping(item) for item in raw_entities),
            source=source,
        )


@dataclass(frozen=True, slots=True)
class Prediction:
    """Финальное предсказание модели для одного документа."""

    hash: str
    entities: tuple[Entity, ...] = ()

    def __post_init__(self) -> None:
        """Сортирует spans и запрещает дубликаты."""
        if not isinstance(self.hash, str) or not self.hash:
            raise ValueError("hash предсказания должен быть непустой строкой")
        ordered = tuple(sorted(self.entities, key=lambda item: (item.start, item.end, item.label)))
        keys = {(entity.label, entity.start, entity.end) for entity in ordered}
        if len(keys) != len(ordered):
            raise ValueError("Предсказание содержит повторяющиеся сущности")
        object.__setattr__(self, "entities", ordered)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Prediction:
        """Создаёт предсказание из записи JSONL."""
        raw_entities = value.get("entities", [])
        if not isinstance(raw_entities, list):
            raise TypeError("entities предсказания должен быть массивом")
        return cls(
            hash=value.get("hash"),
            entities=tuple(Entity.from_mapping(item) for item in raw_entities),
        )

    def to_mapping(self, *, include_scores: bool = False) -> dict[str, Any]:
        """Возвращает предсказание в формате evaluator-а."""
        return {
            "hash": self.hash,
            "entities": [
                entity.to_mapping(include_score=include_scores) for entity in self.entities
            ],
        }
