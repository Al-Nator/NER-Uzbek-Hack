"""Официально совместимые exact-span метрики."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from uzner.domain import LABELS, Document, Entity, Prediction


@dataclass(frozen=True, slots=True)
class Counts:
    """Количество точных совпадений и ошибок."""

    tp: int
    fp: int
    fn: int

    def __post_init__(self) -> None:
        """Запрещает отрицательные счётчики."""
        if min(self.tp, self.fp, self.fn) < 0:
            raise ValueError("Счётчики метрик не могут быть отрицательными")


@dataclass(frozen=True, slots=True)
class Scores:
    """Precision, recall и F1 вместе с исходными счётчиками."""

    precision: float
    recall: float
    f1: float
    counts: Counts

    def to_mapping(self) -> dict[str, float | int]:
        """Возвращает плоский формат, совместимый с JSON-отчётом."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            **asdict(self.counts),
            "gold": self.counts.tp + self.counts.fn,
            "predicted": self.counts.tp + self.counts.fp,
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Полный результат exact-span оценки."""

    records: int
    by_label: dict[str, Scores]
    micro: Scores
    macro_precision: float
    macro_recall: float
    macro_f1: float

    def to_mapping(self) -> dict[str, object]:
        """Сериализует метрики для отчёта запуска."""
        return {
            "schema_version": 1,
            "matching": "same hash and exact label/start/end",
            "records": self.records,
            "by_label": {label: self.by_label[label].to_mapping() for label in LABELS},
            "micro": self.micro.to_mapping(),
            "macro": {
                "precision": self.macro_precision,
                "recall": self.macro_recall,
                "f1": self.macro_f1,
            },
        }


def _entity_key(entity: Entity) -> tuple[str, int, int]:
    """Преобразует сущность в ключ точного сравнения."""
    return entity.label, entity.start, entity.end


def _scores(counts: Counts) -> Scores:
    """Вычисляет precision, recall и F1 из счётчиков."""
    precision = counts.tp / (counts.tp + counts.fp) if counts.tp + counts.fp else 0.0
    recall = counts.tp / (counts.tp + counts.fn) if counts.tp + counts.fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Scores(precision=precision, recall=recall, f1=f1, counts=counts)


def evaluate_predictions(
    gold: Iterable[Document],
    predictions: Iterable[Prediction],
) -> EvaluationResult:
    """Считает exact-span метрики с проверкой полного набора hash."""
    gold_items = list(gold)
    prediction_items = list(predictions)
    if not gold_items:
        raise ValueError("Gold не должен быть пустым")
    gold_by_hash = {document.hash: document for document in gold_items}
    prediction_by_hash = {prediction.hash: prediction for prediction in prediction_items}
    if len(gold_by_hash) != len(gold_items):
        raise ValueError("Gold содержит повторяющиеся hash")
    if len(prediction_by_hash) != len(prediction_items):
        raise ValueError("Предсказания содержат повторяющиеся hash")
    if set(gold_by_hash) != set(prediction_by_hash):
        missing = sorted(set(gold_by_hash) - set(prediction_by_hash))
        unexpected = sorted(set(prediction_by_hash) - set(gold_by_hash))
        raise ValueError(f"Наборы hash различаются: missing={missing}, unexpected={unexpected}")

    mutable_counts = {label: [0, 0, 0] for label in LABELS}
    for hash_value, document in gold_by_hash.items():
        gold_keys = {_entity_key(entity) for entity in document.entities}
        predicted_keys = {_entity_key(entity) for entity in prediction_by_hash[hash_value].entities}
        for label in LABELS:
            expected = {key for key in gold_keys if key[0] == label}
            actual = {key for key in predicted_keys if key[0] == label}
            mutable_counts[label][0] += len(expected & actual)
            mutable_counts[label][1] += len(actual - expected)
            mutable_counts[label][2] += len(expected - actual)

    by_label = {label: _scores(Counts(*mutable_counts[label])) for label in LABELS}
    micro_counts = Counts(
        tp=sum(score.counts.tp for score in by_label.values()),
        fp=sum(score.counts.fp for score in by_label.values()),
        fn=sum(score.counts.fn for score in by_label.values()),
    )
    return EvaluationResult(
        records=len(gold_items),
        by_label=by_label,
        micro=_scores(micro_counts),
        macro_precision=sum(score.precision for score in by_label.values()) / len(LABELS),
        macro_recall=sum(score.recall for score in by_label.values()) / len(LABELS),
        macro_f1=sum(score.f1 for score in by_label.values()) / len(LABELS),
    )
