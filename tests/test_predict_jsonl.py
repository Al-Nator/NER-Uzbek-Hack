"""Контракт файла лидерборда и исходных Unicode offsets."""

import importlib.util

import pytest

from uzner.data.io import read_jsonl, write_predictions
from uzner.domain import Document, Entity, Prediction


def test_submission_contract(tmp_path):
    """Сохраняет только hash/entities; score не попадает в лидерборд."""
    spec = importlib.util.spec_from_file_location("predict_jsonl", "scripts/predict_jsonl.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    docs = (Document("a", "Али"), Document("b", "Нет"))
    predictions = (Prediction("a", (Entity(0, 3, "NAME", 0.9),)), Prediction("b", ()))
    module.validate_predictions(docs, predictions)
    output = tmp_path / "predictions.jsonl"
    write_predictions(output, predictions)
    assert read_jsonl(output) == [
        {"hash": "a", "entities": [{"label": "NAME", "start": 0, "end": 3}]},
        {"hash": "b", "entities": []},
    ]
    with pytest.raises(ValueError):
        module.validate_predictions(docs, predictions[::-1])
    with pytest.raises(ValueError):
        module.validate_predictions(docs[:1], (Prediction("a", (Entity(0, 4, "NAME"),)),))
