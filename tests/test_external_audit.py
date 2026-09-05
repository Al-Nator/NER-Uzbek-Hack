"""Проверки provenance-аудита внешних данных."""

import json
from pathlib import Path

import pytest

from uzner.data.external_audit import audit_directory, audit_external, normalized_text


def test_original_provenance_and_offset_errors_are_not_hidden(tmp_path: Path) -> None:
    """Синтетика под именем real и ошибки поверхности отражаются отдельно."""
    row = {
        "id": "a",
        "text": "Ali Toshkentda",
        "tokens": ["Ali"],
        "ner_tags": [],
        "entities": [
            {"label": "PER", "start": 0, "end": 3, "text": "Vali"},
            {"label": "LOC", "start": -1, "end": 50, "text": "x"},
        ],
        "meta": {"is_synthetic": False, "is_synthetic_original": True},
    }
    path = tmp_path / "uzner_train_bioes.jsonl"
    path.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
    original = path.read_bytes()
    report = audit_external(path, {"dev": {"ali toshkentda"}})
    assert report.records == 2
    assert report.promoted_synthetic == 2
    assert report.synthetic_original == {"true": 2}
    assert report.synthetic_primary == {"false": 2}
    assert report.invalid_offsets == report.surface_mismatch == 2
    assert report.duplicate_ids == report.duplicate_normalized_text == 1
    assert report.official_overlap == {"dev": 2}
    assert report.token_tag_length_mismatch == 2
    assert path.read_bytes() == original


def test_directory_audit_preserves_unknown_provenance(tmp_path: Path) -> None:
    """Отсутствующий original не выдаётся за подтверждённые реальные данные."""
    for split in ("train", "dev"):
        (tmp_path / f"{split}.jsonl").write_text('{"text": "OTHER"}\n')
    with pytest.raises(FileNotFoundError, match="Нет UzNER"):
        audit_directory(tmp_path, tmp_path)
    row = {
        "id": "1",
        "text": "Ali",
        "tokens": ["Ali"],
        "ner_tags": ["S-PER"],
        "entities": [{"label": "PER", "start": 0, "end": 3, "text": "Ali"}],
    }
    (tmp_path / "uzner_train_bioes.jsonl").write_text(json.dumps(row) + "\n")
    result = audit_directory(tmp_path, tmp_path)
    json.dumps(result)
    assert result["files"][0]["synthetic_original"] == {"unknown": 1}
    assert result["files"][0]["invalid_offsets"] == 0
    assert result["files"][0]["surface_mismatch"] == 0
    assert len(result["official_sha256"]["dev"]) == 64


def test_normalization_is_for_comparison_only() -> None:
    """Разные апострофы и пробелы дают одинаковый ключ сравнения."""
    assert normalized_text("  OʻZBEKISTON\n") == "o'zbekiston"
