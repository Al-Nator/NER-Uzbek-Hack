"""Проверки общего инференса и точного контракта ансамблевой посылки."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from tests.test_training_pipeline import _tiny_project
from uzner.config import load_experiment_config
from uzner.data.windows import build_window_features
from uzner.domain import Document, Entity, Prediction
from uzner.models.factory import model_class
from uzner.training import submission
from uzner.training.data_setup import make_loader
from uzner.training.inference import run_inference, run_predictions


@pytest.mark.parametrize("head", ["crf", "global_pointer"])
def test_public_prediction_matches_dev_without_gold(tmp_path: Path, monkeypatch, head: str):
    """CRF и GP используют один декодер; public-инференс не вызывает evaluator."""
    config = load_experiment_config(_tiny_project(tmp_path))
    model_config = replace(
        config.model,
        head=head,
        tag_scheme="bioes",
        architecture="span" if head == "global_pointer" else "token_tagging",
        decoder="span" if head == "global_pointer" else "crf",
    )
    encoder = AutoModel.from_pretrained(tmp_path / "tiny_model")
    tokenizer = AutoTokenizer.from_pretrained(tmp_path / "tiny_model")
    model = model_class(model_config)(encoder, model_config)
    documents = (Document("a", "Ali Toshkent", (Entity(0, 3, "NAME"),)),)
    features = build_window_features(
        documents, tokenizer, config.tokenization, "bioes", with_labels=False
    )
    loader = make_loader(features, tokenizer, batch_size=2, shuffle=False, seed=42, num_workers=0)
    expected = run_inference(
        model, loader, features, documents, (), torch.device("cpu"), bf16=False
    ).predictions

    def forbidden(*args, **kwargs):
        """Останавливает тест при обращении public-инференса к метрикам."""
        pytest.fail("Public inference must not evaluate gold")

    monkeypatch.setattr("uzner.training.inference.evaluate_detailed", forbidden)
    public = (replace(documents[0], entities=()),)
    actual = run_predictions(model, loader, public, torch.device("cpu"), bf16=False)
    assert actual.predictions == expected
    submission.validate_submission(public, actual.predictions)


def test_submission_contract_rejects_bad_inputs():
    """Потерянные hash, дубли и неверные Unicode-границы запрещены."""
    document = Document("a", "Ўз")
    with pytest.raises(ValueError):
        submission.validate_submission((document,), ())
    with pytest.raises(ValueError):
        submission.validate_submission((document, document), ())
    with pytest.raises(ValueError):
        submission.validate_submission((document,), (Prediction("a", (Entity(0, 3, "GEO"),)),))


def test_ensemble_output_and_provenance(tmp_path: Path, monkeypatch):
    """Сохраняются три голоса и manifest, без score и повторной записи результата."""
    ensemble = tmp_path / "runs/s62"
    ensemble.mkdir(parents=True)
    (ensemble / "resolved_config.json").write_text(
        json.dumps({"rule": "exact span 2 of 3", "sources": ["one", "two", "three"]})
    )
    for name in ("one", "two", "three"):
        checkpoint = ensemble.parent / name / "checkpoints/best"
        checkpoint.mkdir(parents=True)
        (checkpoint / "head.safetensors").write_bytes(b"test")
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps({"hash": "a", "text": "Ali"}) + "\n")

    def fake_predict(source, documents, batch_size):
        """Возвращает два согласованных голоса из трёх без модели."""
        entities = () if source.name == "three" else (Entity(0, 3, "NAME"),)
        return (Prediction("a", entities),)

    monkeypatch.setattr(submission, "predict_checkpoint", fake_predict)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test GPU")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    request = submission.EnsembleSubmission(ensemble, input_path, tmp_path / "output")
    output = submission.build_ensemble_submission(request)
    assert json.loads(output.read_text()) == {
        "hash": "a",
        "entities": [{"label": "NAME", "start": 0, "end": 3}],
    }
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert len(manifest["sources"]) == 3
    assert manifest["gold_evaluation"] is False
    assert len(list((output.parent / "components").glob("*.jsonl"))) == 3
    with pytest.raises(FileExistsError):
        submission.build_ensemble_submission(request)
