"""Диагностические exact-span разрезы для Uzbek NER."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from uzner.domain import Document, Entity, Prediction
from uzner.evaluation.errors import ErrorAnalysis, analyze_errors
from uzner.evaluation.metrics import EvaluationResult, evaluate_predictions

EntityPredicate = Callable[[str, Entity], bool]
APOSTROPHES = str.maketrans({"\u2019": "'", "\u02bb": "'", "\u02bc": "'", "`": "'"})
SUFFIX_PATTERN = re.compile(
    r"(?:da|ga|dan|ning|ni|да|га|дан|нинг|ни)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SurfaceIndex:
    """Поверхности train-сущностей для seen/unseen разрезов."""

    exact: frozenset[tuple[str, str]]
    normalized: frozenset[tuple[str, str]]


@dataclass(frozen=True, slots=True)
class SliceResult:
    """Метрики одного именованного разреза."""

    group: str
    name: str
    records: int
    gold_entities: int
    predicted_entities: int
    metrics: EvaluationResult

    def to_mapping(self) -> dict[str, object]:
        """Сериализует один разрез для JSON-отчёта."""
        return {
            "group": self.group,
            "name": self.name,
            "records": self.records,
            "gold_entities": self.gold_entities,
            "predicted_entities": self.predicted_entities,
            "metrics": self.metrics.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class DetailedEvaluation:
    """Полные exact, slice и error-метрики одного checkpoint-а."""

    overall: EvaluationResult
    slices: tuple[SliceResult, ...]
    errors: ErrorAnalysis

    def to_mapping(self) -> dict[str, object]:
        """Сериализует отчёт без тяжёлого error JSONL."""
        grouped: dict[str, dict[str, object]] = {}
        for item in self.slices:
            grouped.setdefault(item.group, {})[item.name] = item.to_mapping()
        return {
            "overall": self.overall.to_mapping(),
            "diagnostics": self.errors.to_mapping(),
            "slices": grouped,
        }


def normalize_surface(value: str) -> str:
    """Нормализует регистр, апострофы и пробелы только для анализа."""
    return " ".join(value.translate(APOSTROPHES).casefold().split())


def detect_script(text: str) -> str:
    """Относит текст к latin, cyrillic, mixed или other."""
    has_latin = False
    has_cyrillic = False
    for character in text:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        has_latin = has_latin or "LATIN" in name
        has_cyrillic = has_cyrillic or "CYRILLIC" in name
    if has_latin and has_cyrillic:
        return "mixed"
    if has_latin:
        return "latin"
    if has_cyrillic:
        return "cyrillic"
    return "other"


def has_attached_suffix(surface: str) -> bool:
    """Эвристически находит узбекский присоединённый суффикс."""
    match = SUFFIX_PATTERN.search(surface)
    return match is not None and match.start() >= 2


def build_surface_index(documents: Sequence[Document]) -> SurfaceIndex:
    """Собирает exact и normalized class-specific поверхности train."""
    exact: set[tuple[str, str]] = set()
    normalized: set[tuple[str, str]] = set()
    for document in documents:
        for entity in document.entities:
            surface = document.text[entity.start : entity.end]
            exact.add((entity.label, surface))
            normalized.add((entity.label, normalize_surface(surface)))
    return SurfaceIndex(frozenset(exact), frozenset(normalized))


def _filter_entities(
    gold: tuple[Document, ...],
    predictions: tuple[Prediction, ...],
    predicate: EntityPredicate,
) -> tuple[tuple[Document, ...], tuple[Prediction, ...]]:
    """Фильтрует gold и prediction spans одним детерминированным правилом."""
    filtered_gold: list[Document] = []
    filtered_predictions: list[Prediction] = []
    for document, prediction in zip(gold, predictions, strict=True):
        filtered_gold.append(
            Document(
                hash=document.hash,
                text=document.text,
                entities=tuple(
                    entity for entity in document.entities if predicate(document.text, entity)
                ),
                source=document.source,
            )
        )
        filtered_predictions.append(
            Prediction(
                hash=prediction.hash,
                entities=tuple(
                    entity for entity in prediction.entities if predicate(document.text, entity)
                ),
            )
        )
    return tuple(filtered_gold), tuple(filtered_predictions)


def _slice_result(
    group: str,
    name: str,
    gold: tuple[Document, ...],
    predictions: tuple[Prediction, ...],
) -> SliceResult:
    """Считает один готовый разрез."""
    return SliceResult(
        group=group,
        name=name,
        records=len(gold),
        gold_entities=sum(len(document.entities) for document in gold),
        predicted_entities=sum(len(prediction.entities) for prediction in predictions),
        metrics=evaluate_predictions(gold, predictions),
    )


def _document_slices(
    gold: tuple[Document, ...], predictions: tuple[Prediction, ...]
) -> list[SliceResult]:
    """Строит разрезы по письменности и длине документа."""
    result: list[SliceResult] = []
    pairs = tuple(zip(gold, predictions, strict=True))
    for name in ("latin", "cyrillic", "mixed", "other"):
        selected = tuple(pair for pair in pairs if detect_script(pair[0].text) == name)
        if selected:
            result.append(
                _slice_result(
                    "script",
                    name,
                    tuple(pair[0] for pair in selected),
                    tuple(pair[1] for pair in selected),
                )
            )
    length_buckets = {
        "short_0_255": lambda size: size < 256,
        "medium_256_1023": lambda size: 256 <= size < 1024,
        "long_1024_plus": lambda size: size >= 1024,
    }
    for name, predicate in length_buckets.items():
        selected = tuple(pair for pair in pairs if predicate(len(pair[0].text)))
        if selected:
            result.append(
                _slice_result(
                    "document_length_chars",
                    name,
                    tuple(pair[0] for pair in selected),
                    tuple(pair[1] for pair in selected),
                )
            )
    return result


def evaluate_detailed(
    gold: tuple[Document, ...],
    predictions: tuple[Prediction, ...],
    train: tuple[Document, ...],
    chunk_edges: dict[str, tuple[int, ...]],
) -> DetailedEvaluation:
    """Считает широкий набор exact-span диагностик."""
    if len(gold) != len(predictions) or not gold:
        raise ValueError("Нужны непустые и равные gold/prediction наборы")
    index = build_surface_index(train)
    slices = _document_slices(gold, predictions)
    predicates: dict[str, dict[str, EntityPredicate]] = {
        "surface": {
            "exact_seen": lambda text, entity: (
                entity.label,
                text[entity.start : entity.end],
            )
            in index.exact,
            "normalized_seen": lambda text, entity: (
                entity.label,
                normalize_surface(text[entity.start : entity.end]),
            )
            in index.normalized
            and (entity.label, text[entity.start : entity.end]) not in index.exact,
            "unseen": lambda text, entity: (
                entity.label,
                normalize_surface(text[entity.start : entity.end]),
            )
            not in index.normalized,
        },
        "span_length_chars": {
            "1_4": lambda _text, entity: entity.end - entity.start <= 4,
            "5_10": lambda _text, entity: 5 <= entity.end - entity.start <= 10,
            "11_20": lambda _text, entity: 11 <= entity.end - entity.start <= 20,
            "21_plus": lambda _text, entity: entity.end - entity.start >= 21,
        },
        "attached_suffix": {
            "yes": lambda text, entity: has_attached_suffix(text[entity.start : entity.end]),
            "no": lambda text, entity: not has_attached_suffix(text[entity.start : entity.end]),
        },
        "entity_script": {
            name: lambda text, entity, script=name: detect_script(text[entity.start : entity.end])
            == script
            for name in ("latin", "cyrillic", "mixed", "other")
        },
        "apostrophe": {
            "yes": lambda text, entity: any(
                mark in text[entity.start : entity.end] for mark in "'\u2019\u02bb\u02bc`"
            ),
            "no": lambda text, entity: not any(
                mark in text[entity.start : entity.end] for mark in "'\u2019\u02bb\u02bc`"
            ),
        },
        "quotes": {
            "yes": lambda text, entity: any(
                mark in text[entity.start : entity.end] for mark in '"“”«»„‟'
            ),
            "no": lambda text, entity: not any(
                mark in text[entity.start : entity.end] for mark in '"“”«»„‟'
            ),
        },
    }
    for group, named in predicates.items():
        for name, predicate in named.items():
            filtered_gold, filtered_predictions = _filter_entities(gold, predictions, predicate)
            slices.append(_slice_result(group, name, filtered_gold, filtered_predictions))
    edge_gold: list[Document] = []
    edge_predictions: list[Prediction] = []
    for document, prediction in zip(gold, predictions, strict=True):
        internal_edges = tuple(
            edge for edge in chunk_edges.get(document.hash, ()) if 0 < edge < len(document.text)
        )

        def near_edge(
            _text: str,
            entity: Entity,
            edges: tuple[int, ...] = internal_edges,
        ) -> bool:
            """Проверяет близость span к внутренней границе окна."""
            return any(min(abs(entity.start - edge), abs(entity.end - edge)) <= 8 for edge in edges)

        filtered_gold, filtered_prediction = _filter_entities((document,), (prediction,), near_edge)
        edge_gold.extend(filtered_gold)
        edge_predictions.extend(filtered_prediction)
    slices.append(
        _slice_result("chunk_edge", "near_8_chars", tuple(edge_gold), tuple(edge_predictions))
    )
    return DetailedEvaluation(
        overall=evaluate_predictions(gold, predictions),
        slices=tuple(slices),
        errors=analyze_errors(gold, predictions),
    )
