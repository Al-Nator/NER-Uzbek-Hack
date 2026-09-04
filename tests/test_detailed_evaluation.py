"""Тесты детальных exact-span разрезов и tokenizer audit."""

import pytest

from tests.helpers import FakeTokenizer, sample_documents
from uzner.config import EncoderConfig, TokenizationConfig
from uzner.domain import Document, Entity, Prediction
from uzner.evaluation.errors import analyze_errors
from uzner.evaluation.slices import (
    build_surface_index,
    detect_script,
    evaluate_detailed,
    has_attached_suffix,
    normalize_surface,
)
from uzner.evaluation.tokenizer_audit import audit_tokenizer


def _evaluation_cases() -> tuple[tuple[Document, ...], tuple[Prediction, ...]]:
    """Создаёт набор, в котором представлены все типы ошибок."""
    gold = (
        Document(
            hash="d1",
            text="Ali ACME Toshkentda",
            entities=(
                Entity(label="NAME", start=0, end=3),
                Entity(label="ORG", start=4, end=8),
                Entity(label="GEO", start=9, end=19),
            ),
            source="dev",
        ),
        Document(
            hash="d2",
            text="Bob Corporation Samarqand extra",
            entities=(
                Entity(label="NAME", start=0, end=3),
                Entity(label="ORG", start=4, end=15),
                Entity(label="GEO", start=16, end=25),
            ),
            source="dev",
        ),
        Document(hash="d3", text="Пустой текст", source="dev"),
        Document(
            hash="d4",
            text="ACME Тошкент",
            entities=(Entity(label="ORG", start=0, end=4),),
            source="dev",
        ),
    )
    predictions = (
        Prediction(
            hash="d1",
            entities=(
                Entity(label="NAME", start=0, end=3),
                Entity(label="GEO", start=4, end=8),
                Entity(label="GEO", start=9, end=17),
            ),
        ),
        Prediction(
            hash="d2",
            entities=(
                Entity(label="NAME", start=0, end=4),
                Entity(label="ORG", start=5, end=16),
                Entity(label="ORG", start=26, end=31),
            ),
        ),
        Prediction(
            hash="d3",
            entities=(Entity(label="NAME", start=0, end=6),),
        ),
        Prediction(
            hash="d4",
            entities=(Entity(label="NAME", start=0, end=3),),
        ),
    )
    return gold, predictions


def test_error_typology_and_secondary_metrics() -> None:
    """Анализ различает границы, класс, пропуск и лишний span."""
    gold, predictions = _evaluation_cases()
    result = analyze_errors(gold, predictions)

    assert result.counts == {
        "boundary_long": 1,
        "boundary_shift": 1,
        "boundary_short": 1,
        "correct": 1,
        "missed": 1,
        "spurious": 2,
        "wrong_label": 1,
        "wrong_label_and_boundary": 1,
    }
    assert result.boundary.tp == 2
    assert result.document_exact_match == 0
    assert result.empty_document_false_positive_rate == 1
    assert 0 < result.mean_matched_iou < 1
    assert len(result.records) == 8
    assert result.records[0].to_mapping()["context"]
    assert result.to_mapping()["boundary"]["f1"] == result.boundary.f1


def test_error_analysis_validates_alignment_and_empty_input() -> None:
    """Несогласованные hash и пустой набор отклоняются."""
    gold, predictions = _evaluation_cases()
    with pytest.raises(ValueError, match="непустые"):
        analyze_errors((), ())
    with pytest.raises(ValueError, match="hash-порядке"):
        analyze_errors(gold[:1], (Prediction(hash="other"),))


def test_surface_helpers_and_detailed_slices() -> None:
    """Разрезы покрывают script, seen, длину, суффикс и chunk edge."""
    gold, predictions = _evaluation_cases()
    train = (
        Document(
            hash="train",
            text="ALI ACME",
            entities=(
                Entity(label="NAME", start=0, end=3),
                Entity(label="ORG", start=4, end=8),
            ),
            source="train",
        ),
    )
    result = evaluate_detailed(
        gold,
        predictions,
        train,
        {"d1": (8,), "d2": (16,), "d3": (), "d4": (4,)},
    )
    grouped = result.to_mapping()["slices"]

    assert normalize_surface("  OʻZ  BEK ") == "o'z bek"
    assert detect_script("Toshkent") == "latin"
    assert detect_script("Тошкент") == "cyrillic"
    assert detect_script("Toshkent Ш") == "mixed"
    assert detect_script("123") == "other"
    assert has_attached_suffix("Toshkentda")
    assert not has_attached_suffix("da")
    index = build_surface_index(train)
    assert ("ORG", "ACME") in index.exact
    assert ("NAME", "ali") in index.normalized
    assert set(grouped) >= {
        "script",
        "document_length_chars",
        "surface",
        "span_length_chars",
        "attached_suffix",
        "chunk_edge",
    }
    assert grouped["surface"]["exact_seen"]["gold_entities"] == 2
    assert grouped["surface"]["normalized_seen"]["gold_entities"] == 1
    assert grouped["surface"]["unseen"]["gold_entities"] == 4
    assert result.overall.micro.f1 > 0

    with pytest.raises(ValueError, match="непустые"):
        evaluate_detailed((), (), train, {})


def test_tokenizer_audit_reports_ceiling_fragmentation_and_slices() -> None:
    """Tokenizer audit показывает exact ceiling и подсловную фрагментацию."""
    documents = (
        *sample_documents(),
        Document(
            hash="partial",
            text="Alice",
            entities=(Entity(label="NAME", start=1, end=4),),
            source="dev",
        ),
        Document(
            hash="apostrophe",
            text="Oʻzbekistonda",
            entities=(Entity(label="GEO", start=0, end=13),),
            source="dev",
        ),
    )
    audit = audit_tokenizer(
        documents,
        FakeTokenizer(),
        EncoderConfig("fake", "revision"),
        TokenizationConfig(max_length=8, stride=2),
    )
    mapping = audit.to_mapping()

    assert audit.overall.entities == 6
    assert audit.overall.representable == 5
    assert audit.overall.representability == pytest.approx(5 / 6)
    assert audit.windows >= len(documents)
    assert mapping["slices"]["apostrophe"]["yes"]["entities"] == 1
    assert mapping["slices"]["attached_suffix"]["yes"]["entities"] >= 1
    assert mapping["slices"]["label"]["GEO"]["p95_subwords"] >= 1
    assert mapping["overall"]["mean_chars_per_subword"] > 0
    assert mapping["slices"]["quotes"]["no"]["entities"] >= 1

    empty_entities = (Document(hash="empty", text="text", source="dev"),)
    empty_audit = audit_tokenizer(
        empty_entities,
        FakeTokenizer(),
        EncoderConfig("fake", "revision"),
        TokenizationConfig(max_length=8, stride=0),
    )
    assert empty_audit.overall.entities == 0
    with pytest.raises(ValueError, match="непустые"):
        audit_tokenizer(
            (),
            FakeTokenizer(),
            EncoderConfig("fake", "revision"),
            TokenizationConfig(),
        )
