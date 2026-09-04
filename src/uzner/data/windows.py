"""Оконная токенизация без потери глобальных character offsets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from uzner.config import TokenizationConfig
from uzner.data.tagging import TagScheme, build_tag_vocabulary, encode_tags
from uzner.domain import Document, Entity

IGNORE_LABEL = -100
Offset = tuple[int, int]


def canonicalize_offsets(text: str, offsets: Sequence[Offset]) -> tuple[Offset, ...]:
    """Убирает внешний пробел tokenizer-а, сохраняя координаты исходной строки."""
    normalized: list[Offset] = []
    for raw_start, raw_end in offsets:
        start, end = int(raw_start), int(raw_end)
        if start == end:
            normalized.append((start, end))
            continue
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        normalized.append((start, end))
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class WindowFeature:
    """Одно токенизированное окно с глобальными offsets."""

    document_index: int
    window_index: int
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_type_ids: tuple[int, ...] | None
    offsets: tuple[Offset, ...]
    labels: tuple[int, ...] | None


@dataclass(slots=True)
class WindowBatch:
    """Дополненный batch окон с метаданными инференса."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None
    labels: torch.Tensor | None
    document_indices: tuple[int, ...]
    window_indices: tuple[int, ...]
    offsets: tuple[tuple[Offset, ...], ...]

    def to(self, device: torch.device) -> WindowBatch:
        """Переносит модельные тензоры batch на устройство."""
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        if self.token_type_ids is not None:
            self.token_type_ids = self.token_type_ids.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
        return self


class WindowDataset(Dataset[WindowFeature]):
    """Набор заранее токенизированных окон."""

    def __init__(self, features: Sequence[WindowFeature]) -> None:
        """Сохраняет непустую последовательность окон."""
        if not features:
            raise ValueError("Набор окон не должен быть пустым")
        self._features = tuple(features)

    def __len__(self) -> int:
        """Возвращает число окон."""
        return len(self._features)

    def __getitem__(self, index: int) -> WindowFeature:
        """Возвращает одно окно по индексу."""
        return self._features[index]


@dataclass(frozen=True, slots=True)
class WindowCollator:
    """Дополняет окна до максимальной длины в batch."""

    pad_token_id: int

    def __call__(self, features: Sequence[WindowFeature]) -> WindowBatch:
        """Собирает один batch тензоров и метаданных."""
        if not features:
            raise ValueError("Batch не должен быть пустым")
        width = max(len(feature.input_ids) for feature in features)
        has_token_types = any(feature.token_type_ids is not None for feature in features)
        has_labels = any(feature.labels is not None for feature in features)
        if has_labels and any(feature.labels is None for feature in features):
            raise ValueError("Нельзя смешивать размеченные и неразмеченные окна")

        def pad(values: Sequence[int], fill: int) -> list[int]:
            """Дополняет одну числовую последовательность."""
            return [*values, *(fill for _ in range(width - len(values)))]

        token_types = None
        if has_token_types:
            token_types = torch.tensor(
                [pad(feature.token_type_ids or (), 0) for feature in features],
                dtype=torch.long,
            )
        labels = None
        if has_labels:
            labels = torch.tensor(
                [pad(feature.labels or (), IGNORE_LABEL) for feature in features],
                dtype=torch.long,
            )
        return WindowBatch(
            input_ids=torch.tensor(
                [pad(feature.input_ids, self.pad_token_id) for feature in features],
                dtype=torch.long,
            ),
            attention_mask=torch.tensor(
                [pad(feature.attention_mask, 0) for feature in features], dtype=torch.long
            ),
            token_type_ids=token_types,
            labels=labels,
            document_indices=tuple(feature.document_index for feature in features),
            window_indices=tuple(feature.window_index for feature in features),
            offsets=tuple(feature.offsets for feature in features),
        )


def _window_bounds(offsets: Sequence[Offset]) -> tuple[int, int]:
    """Возвращает границы содержательной части окна."""
    content = [(start, end) for start, end in offsets if start != end]
    if not content:
        raise ValueError("Окно не содержит токенов текста")
    return content[0][0], content[-1][1]


def _intersects(entity: Entity, offset: Offset) -> bool:
    """Проверяет пересечение сущности и токена."""
    start, end = offset
    return start != end and start < entity.end and end > entity.start


def _ambiguous_entities(
    offsets: Sequence[Offset],
    entities: Sequence[Entity],
) -> frozenset[Entity]:
    """Находит сущности, которые tokenizer объединил в общий токен."""
    ambiguous: set[Entity] = set()
    for offset in offsets:
        claims = [entity for entity in entities if _intersects(entity, offset)]
        if len(claims) > 1:
            ambiguous.update(claims)
    return frozenset(ambiguous)


def align_window_labels(
    offsets: Sequence[Offset],
    entities: Sequence[Entity],
    scheme: TagScheme,
) -> tuple[int, ...]:
    """Кодирует gold и маскирует сущности, обрезанные краем окна."""
    window_start, window_end = _window_bounds(offsets)
    complete = tuple(
        entity for entity in entities if entity.start >= window_start and entity.end <= window_end
    )
    partial = tuple(
        entity
        for entity in entities
        if entity.start < window_end and entity.end > window_start and entity not in complete
    )
    ambiguous = _ambiguous_entities(offsets, complete)
    encodable = tuple(entity for entity in complete if entity not in ambiguous)
    masked_entities = (*partial, *ambiguous)
    vocabulary = build_tag_vocabulary(scheme)
    tag_to_id = {tag: index for index, tag in enumerate(vocabulary)}
    tags = encode_tags(offsets, encodable, scheme)
    labels = [IGNORE_LABEL if tag is None else tag_to_id[tag] for tag in tags]
    for index, offset in enumerate(offsets):
        if any(_intersects(entity, offset) for entity in masked_entities):
            labels[index] = IGNORE_LABEL
    return tuple(labels)


def _chunks(encoded: Any, key: str) -> list[list[Any]] | None:
    """Нормализует одно или несколько окон tokenizer-а."""
    if key not in encoded:
        return None
    values = encoded[key]
    if not values:
        return []
    return values if isinstance(values[0], list) else [values]


def build_window_features(
    documents: Sequence[Document],
    tokenizer: Any,
    tokenization: TokenizationConfig,
    scheme: TagScheme,
    *,
    with_labels: bool,
) -> tuple[WindowFeature, ...]:
    """Токенизирует документы в перекрывающиеся окна."""
    features: list[WindowFeature] = []
    for document_index, document in enumerate(documents):
        encoded = tokenizer(
            document.text,
            truncation=True,
            max_length=tokenization.max_length,
            stride=tokenization.stride,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            padding=False,
        )
        input_chunks = _chunks(encoded, "input_ids") or []
        mask_chunks = _chunks(encoded, "attention_mask") or []
        offset_chunks = _chunks(encoded, "offset_mapping") or []
        type_chunks = _chunks(encoded, "token_type_ids")
        if not (len(input_chunks) == len(mask_chunks) == len(offset_chunks)):
            raise ValueError("Tokenizer вернул несогласованные окна")
        for window_index, (input_ids, attention_mask, raw_offsets) in enumerate(
            zip(input_chunks, mask_chunks, offset_chunks, strict=True)
        ):
            offsets = canonicalize_offsets(document.text, raw_offsets)
            labels = (
                align_window_labels(offsets, document.entities, scheme) if with_labels else None
            )
            token_types = None if type_chunks is None else tuple(type_chunks[window_index])
            features.append(
                WindowFeature(
                    document_index=document_index,
                    window_index=window_index,
                    input_ids=tuple(input_ids),
                    attention_mask=tuple(attention_mask),
                    token_type_ids=token_types,
                    offsets=offsets,
                    labels=labels,
                )
            )
    return tuple(features)


def collect_chunk_edges(features: Sequence[WindowFeature]) -> dict[int, tuple[int, ...]]:
    """Собирает внутренние character-границы окон по документам."""
    grouped: dict[int, set[int]] = {}
    for feature in features:
        start, end = _window_bounds(feature.offsets)
        grouped.setdefault(feature.document_index, set()).update({start, end})
    return {index: tuple(sorted(edges - {0})) for index, edges in grouped.items()}
