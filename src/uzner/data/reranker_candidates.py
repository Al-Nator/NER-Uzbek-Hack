"""Gold-free кандидаты, признаки согласия и символьный контекст для reranker."""

import math
from dataclasses import dataclass

from uzner.domain import LABELS, Document, Entity, Prediction
from uzner.training.submission import validate_submission


@dataclass(frozen=True)
class SpanCandidate:
    """Один предложенный span; gold-метки не входят в признаки."""

    document_index: int
    entity: Entity
    features: tuple[float, ...]
    characters: tuple[int, ...]


def candidates_from_sources(
    documents: tuple[Document, ...], sources: tuple[tuple[Prediction, ...], ...]
) -> tuple[SpanCandidate, ...]:
    """Создаёт объединение exact-spans трёх систем, не подмешивая gold."""
    if len(sources) != 3:
        raise ValueError("Нужны ровно три источника кандидатов")
    for source in sources:
        validate_submission(documents, source)
    result = []
    for index, document in enumerate(documents):
        votes = [{(e.label, e.start, e.end) for e in source[index].entities} for source in sources]
        for label, start, end in sorted(set.union(*votes)):
            span = document.text[start:end]
            left, right = document.text[max(0, start - 32) : start], document.text[end : end + 32]
            agreement = tuple(float((label, start, end) in v) for v in votes)
            features = (
                *[float(label == item) for item in LABELS],
                *agreement,
                sum(agreement) / 3,
                math.log1p(len(span)),
                math.log1p(len(span.split())),
                math.log1p(len(document.text)),
                start / max(1, len(document.text)),
                sum(c.isupper() for c in span) / len(span),
                sum("а" <= c.lower() <= "я" or c in "ўқғҳЎҚҒҲ" for c in span) / len(span),
                sum("a" <= c.lower() <= "z" for c in span) / len(span),
                sum(c.isdigit() for c in span) / len(span),
                float(bool(left) and left[-1].isalnum()),
                float(bool(right) and right[0].isalnum()),
                float(span[0].isalnum()),
                float(span[-1].isalnum()),
                math.log1p(document.text.count(span)),
            )
            # Отдельные маркеры границ; хэш Unicode не изменяет исходные offsets.
            chars = [ord(c) % 4092 + 4 for c in left] + [2]
            chars += [ord(c) % 4092 + 4 for c in span[:64]] + [3]
            chars += [ord(c) % 4092 + 4 for c in right]
            result.append(
                SpanCandidate(
                    index,
                    Entity(start, end, label),
                    features,
                    tuple(chars + [0] * (130 - len(chars))),
                )
            )
    return tuple(result)


def candidate_targets(
    candidates: tuple[SpanCandidate, ...], gold: tuple[Document, ...]
) -> tuple[float, ...]:
    """Добавляет binary exact-span цели только после построения кандидатов."""
    expected = [{(e.label, e.start, e.end) for e in d.entities} for d in gold]
    return tuple(
        float((c.entity.label, c.entity.start, c.entity.end) in expected[c.document_index])
        for c in candidates
    )
