"""Типология exact-span ошибок для точечного анализа."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from uzner.domain import Document, Entity, Prediction


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    """Одна выровненная ошибка с контекстом."""

    hash: str
    kind: str
    gold: Entity | None
    predicted: Entity | None
    context: str

    def to_mapping(self) -> dict[str, object]:
        """Сериализует ошибку для JSONL-отчёта."""
        return {
            "hash": self.hash,
            "kind": self.kind,
            "gold": None if self.gold is None else self.gold.to_mapping(),
            "predicted": None if self.predicted is None else self.predicted.to_mapping(),
            "context": self.context,
        }


@dataclass(frozen=True, slots=True)
class BoundaryScores:
    """Метрики точных границ без учёта класса."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


@dataclass(frozen=True, slots=True)
class ErrorAnalysis:
    """Сводка типов ошибок и детальные записи."""

    counts: dict[str, int]
    boundary: BoundaryScores
    document_exact_match: float
    empty_document_false_positive_rate: float
    mean_matched_iou: float
    records: tuple[ErrorRecord, ...]

    def to_mapping(self) -> dict[str, object]:
        """Сериализует сводку ошибок без тяжёлого списка примеров."""
        return {
            "counts": self.counts,
            "boundary": asdict(self.boundary),
            "document_exact_match": self.document_exact_match,
            "empty_document_false_positive_rate": self.empty_document_false_positive_rate,
            "mean_matched_iou": self.mean_matched_iou,
        }


def _span_iou(left: Entity, right: Entity) -> float:
    """Считает 1D IoU двух character spans."""
    intersection = max(0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union else 0.0


def _context(text: str, gold: Entity | None, predicted: Entity | None) -> str:
    """Вырезает короткий контекст вокруг ошибки."""
    entities = [entity for entity in (gold, predicted) if entity is not None]
    start = max(0, min(entity.start for entity in entities) - 40)
    end = min(len(text), max(entity.end for entity in entities) + 40)
    return text[start:end]


def _error_kind(gold: Entity, predicted: Entity) -> str:
    """Классифицирует лучшую пересекающуюся пару."""
    if gold.start == predicted.start and gold.end == predicted.end:
        return "wrong_label"
    if gold.label != predicted.label:
        return "wrong_label_and_boundary"
    if predicted.start >= gold.start and predicted.end <= gold.end:
        return "boundary_short"
    if predicted.start <= gold.start and predicted.end >= gold.end:
        return "boundary_long"
    return "boundary_shift"


def _boundary_scores(
    gold: tuple[Document, ...],
    predictions: tuple[Prediction, ...],
) -> BoundaryScores:
    """Считает exact-boundary F1 без учёта entity label."""
    tp = fp = fn = 0
    for document, prediction in zip(gold, predictions, strict=True):
        expected = {(entity.start, entity.end) for entity in document.entities}
        actual = {(entity.start, entity.end) for entity in prediction.entities}
        tp += len(expected & actual)
        fp += len(actual - expected)
        fn += len(expected - actual)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BoundaryScores(precision, recall, f1, tp, fp, fn)


def analyze_errors(
    gold: tuple[Document, ...],
    predictions: tuple[Prediction, ...],
) -> ErrorAnalysis:
    """Сопоставляет spans по overlap и строит типологию ошибок."""
    if len(gold) != len(predictions) or not gold:
        raise ValueError("Нужны непустые и равные наборы gold и predictions")
    counts: Counter[str] = Counter()
    records: list[ErrorRecord] = []
    matched_ious: list[float] = []
    exact_documents = 0
    empty_documents = empty_with_fp = 0
    for document, prediction in zip(gold, predictions, strict=True):
        if document.hash != prediction.hash:
            raise ValueError("Gold и predictions должны идти в одинаковом hash-порядке")
        expected = {(e.label, e.start, e.end) for e in document.entities}
        actual = {(e.label, e.start, e.end) for e in prediction.entities}
        exact_documents += expected == actual
        if not document.entities:
            empty_documents += 1
            empty_with_fp += bool(prediction.entities)

        unmatched = list(prediction.entities)
        for entity in document.entities:
            exact = next(
                (
                    candidate
                    for candidate in unmatched
                    if (candidate.label, candidate.start, candidate.end)
                    == (entity.label, entity.start, entity.end)
                ),
                None,
            )
            if exact is not None:
                unmatched.remove(exact)
                counts["correct"] += 1
                matched_ious.append(1.0)
                continue
            overlapping = [candidate for candidate in unmatched if _span_iou(entity, candidate) > 0]
            if not overlapping:
                counts["missed"] += 1
                records.append(
                    ErrorRecord(
                        document.hash,
                        "missed",
                        entity,
                        None,
                        _context(document.text, entity, None),
                    )
                )
                continue
            candidate = max(overlapping, key=lambda item: _span_iou(entity, item))
            unmatched.remove(candidate)
            kind = _error_kind(entity, candidate)
            counts[kind] += 1
            matched_ious.append(_span_iou(entity, candidate))
            records.append(
                ErrorRecord(
                    document.hash,
                    kind,
                    entity,
                    candidate,
                    _context(document.text, entity, candidate),
                )
            )
        for candidate in unmatched:
            counts["spurious"] += 1
            records.append(
                ErrorRecord(
                    document.hash,
                    "spurious",
                    None,
                    candidate,
                    _context(document.text, None, candidate),
                )
            )
    return ErrorAnalysis(
        counts=dict(sorted(counts.items())),
        boundary=_boundary_scores(gold, predictions),
        document_exact_match=exact_documents / len(gold),
        empty_document_false_positive_rate=(
            empty_with_fp / empty_documents if empty_documents else 0.0
        ),
        mean_matched_iou=sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
        records=tuple(records),
    )
