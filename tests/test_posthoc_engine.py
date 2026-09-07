"""Проверки полной матрицы, отсутствия leakage и воспроизводимой статистики."""

from dataclasses import replace
from pathlib import Path

import pytest

from uzner.domain import Document, Entity, Prediction
from uzner.posthoc.config import Variant
from uzner.posthoc.engine import Inputs, apply_variant
from uzner.posthoc.lexicon import build_lexicon
from uzner.posthoc.reporting import paired_bootstrap, save_changes, split_metrics
from uzner.posthoc.study import Study


@pytest.fixture
def inputs():
    """Собирает контролируемые предсказания трёх разных источников."""
    documents = (Document("d1", "Ali Ali"), Document("d2", "Toshkent"))
    source = (Prediction("d1", (Entity(0, 3, "NAME"),)), Prediction("d2", (Entity(0, 8, "GEO"),)))
    train = tuple(Document(str(i), "Ali", (Entity(0, 3, "NAME"),)) for i in range(4))
    candidates = tuple(
        Prediction(p.hash, tuple(replace(e, score=0.9) for e in p.entities)) for p in source
    )
    sources = {name: source for name in ("s33", "s21", "s31", "s62", "s45", "s48")}
    return Inputs(
        documents, sources, candidates, build_lexicon(train), build_lexicon(train, True), 0.05
    )


@pytest.mark.parametrize(
    "operation",
    [
        "identity",
        "repeat",
        "boundary",
        "lex_add",
        "lex_relabel",
        "short",
        "confirm",
        "vote",
        "lex_confirm",
        "gp",
        "gp_vote",
    ],
)
def test_operations_flat_and_gold_free(inputs, operation):
    """Каждая операция возвращает полный плоский ответ с исходными координатами."""
    result = apply_variant(
        inputs, Variant("variant", operation, repeat_after=True, boundary_after=True)
    )
    assert tuple(p.hash for p in result) == ("d1", "d2")
    for doc, pred in zip(inputs.documents, result, strict=True):
        Document(doc.hash, doc.text, pred.entities)


def test_engine_rejects_gold_low_floor_and_order(inputs):
    """Ошибки provenance и gold leakage прерывают запуск до публикации результата."""
    gold = (Document("d1", "Ali Ali", (Entity(0, 3, "NAME"),)), inputs.documents[1])
    with pytest.raises(ValueError, match="gold"):
        apply_variant(replace(inputs, documents=gold), Variant("v"))
    with pytest.raises(ValueError, match="floor"):
        apply_variant(inputs, Variant("v", "gp", threshold=0.01))
    sources = {**inputs.sources, "s62": tuple(reversed(inputs.sources["s62"]))}
    with pytest.raises(ValueError, match="порядок"):
        apply_variant(replace(inputs, sources=sources), Variant("v"))


@pytest.mark.parametrize(
    "values",
    [
        {"name": "../bad"},
        {"operation": "typo"},
        {"selection": "bad"},
        {"labels": ("PER",)},
        {"min_length": 0},
        {"propensity": 1.1},
        {"boundary": "bad"},
    ],
)
def test_variant_validation(values):
    """Опечатки, пути и невозможные параметры не проходят в конфигурацию."""
    with pytest.raises(ValueError):
        Variant(**{"name": "test", **values})


def test_declared_matrix_has_twenty_nontrivial_variants():
    """Зафиксированная матрица содержит больше двадцати проверок, не считая контролей."""
    study = Study.load(Path("configs/posthoc/decoding_v1.yaml"))
    assert sum(v.operation != "identity" for v in study.variants) >= 20
    assert len({v.operation for v in study.variants}) >= 8


def test_changes_halves_and_paired_bootstrap(inputs, tmp_path):
    """Аудит различает добавленный TP и FP, а бутстрэп воспроизводим по seed."""
    old = inputs.sources["s62"]
    gold = (
        Document("d1", "Ali Ali", (Entity(0, 3, "NAME"), Entity(4, 7, "NAME"))),
        Document("d2", "Toshkent", (Entity(0, 8, "GEO"),)),
    )
    new = apply_variant(inputs, Variant("r", "repeat", min_length=2))
    changes = save_changes(tmp_path, gold, old, new)
    assert changes.added_tp == 1 and changes.added_fp == 0 and changes.changed_documents == 1
    halves = split_metrics(gold, new)
    assert halves["half_a"]["micro"]["f1"] == 1
    bootstrap = paired_bootstrap(gold, old, new, samples=100)
    assert bootstrap == paired_bootstrap(gold, old, new, samples=100)
    assert not bootstrap["selection_bias_corrected"]
    assert paired_bootstrap(gold, old, old, samples=10)["delta_ci95"] == [0.0, 0.0]
