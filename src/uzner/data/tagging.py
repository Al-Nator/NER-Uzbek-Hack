"""Преобразования между character spans и BIO/BIOES-разметкой."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from uzner.domain import LABELS, Entity

TagScheme = Literal["bio", "bioes"]
Offset = tuple[int, int]


def validate_scheme(scheme: str) -> TagScheme:
    """Проверяет и уточняет тип схемы тегов."""
    if scheme not in {"bio", "bioes"}:
        raise ValueError(f"Поддерживаются только bio и bioes, получено {scheme!r}")
    return scheme


def build_tag_vocabulary(scheme: TagScheme) -> tuple[str, ...]:
    """Строит детерминированный словарь тегов для указанной схемы."""
    validate_scheme(scheme)
    prefixes = ("B", "I") if scheme == "bio" else ("B", "I", "E", "S")
    return ("O", *(f"{prefix}-{label}" for label in LABELS for prefix in prefixes))


def _entity_token_indices(offsets: Sequence[Offset], entity: Entity) -> list[int]:
    """Находит токены, пересекающие заданную сущность."""
    return [
        index
        for index, (start, end) in enumerate(offsets)
        if start != end and start < entity.end and end > entity.start
    ]


def is_exactly_representable(offsets: Sequence[Offset], entity: Entity) -> bool:
    """Проверяет, совпадают ли границы сущности с границами токенов."""
    indices = _entity_token_indices(offsets, entity)
    if not indices:
        return False
    return offsets[indices[0]][0] == entity.start and offsets[indices[-1]][1] == entity.end


def _prefixes_for_length(length: int, scheme: TagScheme) -> tuple[str, ...]:
    """Возвращает последовательность префиксов для одной сущности."""
    if length < 1:
        raise ValueError("Сущность должна покрывать хотя бы один токен")
    if scheme == "bio":
        return ("B", *("I" for _ in range(length - 1)))
    if length == 1:
        return ("S",)
    return ("B", *("I" for _ in range(length - 2)), "E")


def encode_tags(
    offsets: Sequence[Offset],
    entities: Sequence[Entity],
    scheme: TagScheme,
) -> tuple[str | None, ...]:
    """Кодирует spans в теги; special-токены обозначает значением None."""
    validate_scheme(scheme)
    tags: list[str | None] = [None if start == end else "O" for start, end in offsets]
    for entity in entities:
        indices = _entity_token_indices(offsets, entity)
        if not indices:
            continue
        for index, prefix in zip(indices, _prefixes_for_length(len(indices), scheme), strict=True):
            if tags[index] != "O":
                raise ValueError("Две сущности претендуют на один токен")
            tags[index] = f"{prefix}-{entity.label}"
    return tuple(tags)


def _split_tag(tag: str) -> tuple[str, str | None]:
    """Разделяет тег на префикс и класс сущности."""
    if tag == "O":
        return "O", None
    prefix, separator, label = tag.partition("-")
    if separator != "-" or label not in LABELS:
        raise ValueError(f"Некорректный тег: {tag!r}")
    return prefix, label


def _decode_bio(offsets: Sequence[Offset], tags: Sequence[str | None]) -> tuple[Entity, ...]:
    """Строго декодирует валидную BIO-последовательность."""
    entities: list[Entity] = []
    current_label: str | None = None
    current_start = -1
    current_end = -1

    def flush() -> None:
        """Добавляет накопленную BIO-сущность в результат."""
        nonlocal current_label, current_start, current_end
        if current_label is not None:
            entities.append(Entity(label=current_label, start=current_start, end=current_end))
        current_label = None
        current_start = -1
        current_end = -1

    for offset, tag in zip(offsets, tags, strict=True):
        if tag is None:
            continue
        prefix, label = _split_tag(tag)
        if prefix == "O":
            flush()
        elif prefix == "B":
            flush()
            current_label = label
            current_start, current_end = offset
        elif prefix == "I" and current_label == label:
            current_end = offset[1]
        else:
            raise ValueError(f"Невалидный BIO-переход к тегу {tag!r}")
    flush()
    return tuple(entities)


def _decode_bioes(offsets: Sequence[Offset], tags: Sequence[str | None]) -> tuple[Entity, ...]:
    """Строго декодирует валидную BIOES-последовательность."""
    entities: list[Entity] = []
    current_label: str | None = None
    current_start = -1
    for offset, tag in zip(offsets, tags, strict=True):
        if tag is None:
            continue
        prefix, label = _split_tag(tag)
        if prefix == "O":
            if current_label is not None:
                raise ValueError("Незакрытая BIOES-сущность перед O")
        elif prefix == "S":
            if current_label is not None:
                raise ValueError("S-тег встретился внутри BIOES-сущности")
            entities.append(Entity(label=label, start=offset[0], end=offset[1]))
        elif prefix == "B":
            if current_label is not None:
                raise ValueError("B-тег встретился до закрытия BIOES-сущности")
            current_label = label
            current_start = offset[0]
        elif prefix == "I" and current_label == label:
            continue
        elif prefix == "E" and current_label == label:
            entities.append(Entity(label=label, start=current_start, end=offset[1]))
            current_label = None
            current_start = -1
        else:
            raise ValueError(f"Невалидный BIOES-переход к тегу {tag!r}")
    if current_label is not None:
        raise ValueError("BIOES-последовательность закончилась без E-тега")
    return tuple(entities)


def decode_tags(
    offsets: Sequence[Offset],
    tags: Sequence[str | None],
    scheme: TagScheme,
) -> tuple[Entity, ...]:
    """Декодирует валидную последовательность тегов в character spans."""
    validate_scheme(scheme)
    if len(offsets) != len(tags):
        raise ValueError("Количество offsets и тегов должно совпадать")
    if scheme == "bio":
        return _decode_bio(offsets, tags)
    return _decode_bioes(offsets, tags)
