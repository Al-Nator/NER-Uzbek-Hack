"""Тесты exact-span evaluator-а."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from uzner.domain import Document, Entity, Prediction
from uzner.evaluation.metrics import Counts, evaluate_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_official_evaluator() -> ModuleType:
    """Загружает reference evaluator организаторов без изменения его кода."""
    path = PROJECT_ROOT / "ner_uz_hackathon_participant/scripts/evaluate.py"
    spec = importlib.util.spec_from_file_location("official_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не удалось загрузить evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_match_counts_partial_boundary_as_fp_and_fn() -> None:
    """Частично совпавшая граница не получает частичного зачёта."""
    gold = [
        Document(
            hash="a",
            text="Toshkentda",
            entities=(Entity(label="GEO", start=0, end=10),),
            source="gold",
        )
    ]
    predictions = [Prediction(hash="a", entities=(Entity(label="GEO", start=0, end=8),))]

    result = evaluate_predictions(gold, predictions)

    assert result.by_label["GEO"].counts == Counts(tp=0, fp=1, fn=1)
    assert result.micro.f1 == 0.0


def test_perfect_prediction_and_serialization() -> None:
    """Полное совпадение даёт единичные micro-метрики и JSON-формат."""
    entity = Entity(label="NAME", start=0, end=3)
    result = evaluate_predictions(
        [Document(hash="a", text="Ali", entities=(entity,), source="gold")],
        [Prediction(hash="a", entities=(entity,))],
    )

    assert result.micro.f1 == 1.0
    report = result.to_mapping()
    assert report["schema_version"] == 1
    assert report["records"] == 1
    assert report["matching"] == "same hash and exact label/start/end"
    assert report["micro"]["gold"] == 1
    assert report["micro"]["predicted"] == 1


def test_serialized_metrics_match_official_reference() -> None:
    """Наш evaluator возвращает точно тот же JSON-отчёт, что и official."""
    gold_entities = {
        ("NAME", 0, 3),
        ("GEO", 4, 12),
    }
    predicted_entities = {
        ("NAME", 0, 3),
        ("ORG", 4, 12),
    }
    official = _load_official_evaluator().calculate_metrics(
        {"a": {"text": "Ali Toshkent", "entities": gold_entities}},
        {"a": predicted_entities},
    )
    ours = evaluate_predictions(
        [
            Document(
                hash="a",
                text="Ali Toshkent",
                entities=(
                    Entity(label="NAME", start=0, end=3),
                    Entity(label="GEO", start=4, end=12),
                ),
                source="gold",
            )
        ],
        [
            Prediction(
                hash="a",
                entities=(
                    Entity(label="NAME", start=0, end=3),
                    Entity(label="ORG", start=4, end=12),
                ),
            )
        ],
    ).to_mapping()

    assert ours == official


def test_hash_sets_and_duplicates_are_validated() -> None:
    """Evaluator требует ровно одно предсказание на каждый документ."""
    document = Document(hash="a", text="x", source="gold")
    prediction = Prediction(hash="a")
    with pytest.raises(ValueError, match="Gold"):
        evaluate_predictions([], [])
    with pytest.raises(ValueError, match="Gold"):
        evaluate_predictions([document, document], [prediction])
    with pytest.raises(ValueError, match="Предсказания"):
        evaluate_predictions([document], [prediction, prediction])
    with pytest.raises(ValueError, match="Наборы hash"):
        evaluate_predictions([document], [Prediction(hash="b")])


def test_negative_counts_are_rejected() -> None:
    """Внутренние счётчики не допускают отрицательных значений."""
    with pytest.raises(ValueError):
        Counts(tp=-1, fp=0, fn=0)
