"""Тесты JSONL и объединения источников."""

import json
from pathlib import Path

import pytest

from uzner.data.io import load_documents, read_jsonl, sha256_file, write_predictions
from uzner.domain import Entity, Prediction


def write_jsonl(path: Path, records: list[object]) -> None:
    """Записывает тестовый JSONL без скрытых преобразований."""
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_load_multiple_sources_and_hash_file(tmp_path: Path) -> None:
    """Несколько файлов объединяются с сохранением provenance."""
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl(first, [{"hash": "a", "text": "Ali", "entities": []}])
    write_jsonl(second, [{"hash": "b", "text": "Toshkent", "entities": []}])

    documents = load_documents((("first", first), ("second", second)))

    assert [document.source for document in documents] == ["first", "second"]
    assert len(sha256_file(first)) == 64


def test_duplicate_hash_between_sources_is_error(tmp_path: Path) -> None:
    """Повторяющийся hash между будущими источниками запрещён."""
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    record = {"hash": "same", "text": "x", "entities": []}
    write_jsonl(first, [record])
    write_jsonl(second, [record])

    with pytest.raises(ValueError, match="повторяется"):
        load_documents((("first", first), ("second", second)))


def test_read_jsonl_rejects_blank_invalid_and_non_object(tmp_path: Path) -> None:
    """Парсер сообщает об основных нарушениях формата JSONL."""
    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="пустая"):
        read_jsonl(blank)

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("{\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        read_jsonl(invalid)

    scalar = tmp_path / "scalar.jsonl"
    scalar.write_text("1\n", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON-объект"):
        read_jsonl(scalar)


def test_write_predictions_is_compact_and_removes_temporary_file(tmp_path: Path) -> None:
    """Предсказания записываются атомарно в ожидаемом формате."""
    output = tmp_path / "nested" / "predictions.jsonl"
    prediction = Prediction(
        hash="a",
        entities=(Entity(label="NAME", start=0, end=3, score=0.9),),
    )

    write_predictions(output, [prediction])

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "hash": "a",
        "entities": [{"label": "NAME", "start": 0, "end": 3}],
    }
    assert not output.with_suffix(".jsonl.tmp").exists()
