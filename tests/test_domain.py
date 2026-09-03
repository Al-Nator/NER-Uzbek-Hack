"""Тесты строгих доменных типов."""

import pytest

from uzner.domain import Document, Entity, Prediction


def test_entity_roundtrip_with_optional_score() -> None:
    """Сущность сохраняет поля и управляемо включает score."""
    entity = Entity.from_mapping({"label": "GEO", "start": 4, "end": 12, "score": 0.9})

    assert entity.to_mapping() == {"label": "GEO", "start": 4, "end": 12}
    assert entity.to_mapping(include_score=True)["score"] == 0.9


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"label": "OTHER", "start": 0, "end": 1}, ValueError),
        ({"label": "GEO", "start": True, "end": 1}, TypeError),
        ({"label": "GEO", "start": 0, "end": False}, TypeError),
        ({"label": "GEO", "start": -1, "end": 1}, ValueError),
        ({"label": "GEO", "start": 1, "end": 1}, ValueError),
        ({"label": "GEO", "start": 0, "end": 1, "score": float("nan")}, ValueError),
    ],
)
def test_entity_rejects_invalid_values(
    kwargs: dict[str, object], error_type: type[Exception]
) -> None:
    """Сущность отклоняет неизвестные классы и некорректные границы."""
    with pytest.raises(error_type):
        Entity(**kwargs)


def test_document_sorts_entities_and_preserves_source() -> None:
    """Документ сортирует непересекающиеся spans без изменения текста."""
    document = Document(
        hash="doc-1",
        text="Ali Toshkent",
        entities=(
            Entity(label="GEO", start=4, end=12),
            Entity(label="NAME", start=0, end=3),
        ),
        source="fixture",
    )

    assert [entity.label for entity in document.entities] == ["NAME", "GEO"]
    assert document.source == "fixture"


@pytest.mark.parametrize(
    "document",
    [
        {"hash": "", "text": "x", "entities": (), "source": "fixture"},
        {"hash": "x", "text": 1, "entities": (), "source": "fixture"},
        {"hash": "x", "text": "x", "entities": (), "source": ""},
        {
            "hash": "x",
            "text": "x",
            "entities": (Entity(label="ORG", start=0, end=2),),
            "source": "fixture",
        },
        {
            "hash": "x",
            "text": "abcd",
            "entities": (
                Entity(label="ORG", start=0, end=3),
                Entity(label="GEO", start=2, end=4),
            ),
            "source": "fixture",
        },
    ],
)
def test_document_rejects_invalid_records(document: dict[str, object]) -> None:
    """Документ отклоняет пустые поля, выход за текст и пересечения."""
    with pytest.raises((TypeError, ValueError)):
        Document(**document)


def test_document_from_mapping_requires_entity_array() -> None:
    """JSON-представление документа требует массив entities."""
    with pytest.raises(TypeError):
        Document.from_mapping({"hash": "x", "text": "x", "entities": {}}, source="s")


def test_prediction_rejects_duplicate_and_invalid_mapping() -> None:
    """Финальное предсказание не допускает повторяющиеся spans."""
    entity = Entity(label="NAME", start=0, end=3)
    with pytest.raises(ValueError):
        Prediction(hash="x", entities=(entity, entity))
    with pytest.raises(TypeError):
        Prediction.from_mapping({"hash": "x", "entities": {}})
    with pytest.raises(ValueError):
        Prediction(hash="")


def test_prediction_roundtrip() -> None:
    """Предсказание сериализуется в формат evaluator-а."""
    prediction = Prediction.from_mapping(
        {"hash": "x", "entities": [{"label": "NAME", "start": 0, "end": 3}]}
    )

    assert prediction.to_mapping() == {
        "hash": "x",
        "entities": [{"label": "NAME", "start": 0, "end": 3}],
    }
