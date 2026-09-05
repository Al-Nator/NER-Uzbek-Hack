"""Разреженные gold-пары и общий collator span-моделей."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

from uzner.data.windows import Offset, WindowBatch, WindowCollator, WindowFeature
from uzner.domain import LABELS, Document


@dataclass(frozen=True, slots=True)
class SpanFeature(WindowFeature):
    """Окно с разреженными целевыми парами и исключаемыми токенами."""

    span_targets: tuple[tuple[int, int, int], ...] = ()
    ignored_tokens: tuple[int, ...] = ()


def boundary_indices(offsets: Sequence[Offset]) -> tuple[dict[int, int], dict[int, int]]:
    """Выбирает первый start и последний end при повторных offsets."""
    starts, ends = {}, {}
    for index, (start, end) in enumerate(offsets):
        if start < end:
            starts.setdefault(start, index)
            ends[end] = index
    return starts, ends


def span_pair_mask(offsets: Sequence[Offset], width: int) -> torch.Tensor:
    """Разрешает содержательные упорядоченные пары без ограничения длины."""
    starts, ends = boundary_indices(offsets)
    left = torch.zeros(width, dtype=torch.bool)
    right = torch.zeros(width, dtype=torch.bool)
    left[list(starts.values())] = True
    right[list(ends.values())] = True
    mask = torch.triu(left[:, None] & right[None, :])
    # Нестандартные tokenizer offsets не должны образовать обратный char-span.
    raw = torch.zeros((width, 2), dtype=torch.long)
    raw[: len(offsets)] = torch.tensor(offsets, dtype=torch.long).reshape(-1, 2)
    return mask & (raw[:, 0, None] < raw[None, :, 1])


def with_span_targets(
    features: tuple[WindowFeature, ...], documents: tuple[Document, ...]
) -> tuple[SpanFeature, ...]:
    """Привязывает точные gold-пары, маскируя частичные и непредставимые сущности."""
    result = []
    for feature in features:
        starts, ends = boundary_indices(feature.offsets)
        targets, ignored = [], set()
        for entity in documents[feature.document_index].entities:
            start, end = starts.get(entity.start), ends.get(entity.end)
            if start is not None and end is not None and start <= end:
                targets.append((LABELS.index(entity.label), start, end))
            else:
                ignored.update(
                    index
                    for index, (left, right) in enumerate(feature.offsets)
                    if left < right and left < entity.end and right > entity.start
                )
        result.append(
            SpanFeature(
                feature.document_index,
                feature.window_index,
                feature.input_ids,
                feature.attention_mask,
                feature.token_type_ids,
                feature.offsets,
                None,
                tuple(targets),
                tuple(sorted(ignored)),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SpanCollator:
    """Создаёт плотные targets только для текущего batch, не для всего корпуса."""

    pad_token_id: int

    def __call__(self, features: Sequence[SpanFeature]) -> WindowBatch:
        """Собирает batch × classes × start × end с маской -100."""
        batch = WindowCollator(self.pad_token_id)(features)
        width = batch.input_ids.shape[1]
        labels = torch.full((len(features), len(LABELS), width, width), -100.0)
        for row, feature in enumerate(features):
            valid = span_pair_mask(feature.offsets, width)
            ignored = torch.zeros(width, dtype=torch.long)
            ignored[list(feature.ignored_tokens)] = 1
            prefix = torch.nn.functional.pad(ignored.cumsum(0), (1, 0))
            # Исключаем также пары, охватывающие неизвестную сущность внутри.
            valid &= (prefix[1:][None, :] - prefix[:-1][:, None]) == 0
            labels[row, :, valid] = 0
            for label, start, end in feature.span_targets:
                if valid[start, end]:
                    labels[row, label, start, end] = 1
        return replace(batch, labels=labels)


def span_coverage(
    features: tuple[SpanFeature, ...], documents: tuple[Document, ...]
) -> dict[str, int | float]:
    """Считает покрытие gold точными парами во всех окнах до обучения."""
    represented = set()
    for feature in features:
        for label, start, end in feature.span_targets:
            if not any(start <= index <= end for index in feature.ignored_tokens):
                represented.add(
                    (
                        feature.document_index,
                        LABELS[label],
                        feature.offsets[start][0],
                        feature.offsets[end][1],
                    )
                )
    total = sum(len(document.entities) for document in documents)
    return {
        "gold": total,
        "represented": len(represented),
        "recall_ceiling": len(represented) / total if total else 1.0,
    }
