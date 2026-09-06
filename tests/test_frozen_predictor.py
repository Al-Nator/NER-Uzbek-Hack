"""Проверка чтения реального локального GP-checkpoint без сети/GPU."""

import json
from contextlib import nullcontext

import torch
import yaml

from tests.test_training_pipeline import _tiny_project
from uzner.config import load_experiment_config
from uzner.data.windows import WindowBatch
from uzner.domain import Document, Prediction
from uzner.training import frozen
from uzner.training.checkpoint import load_model_checkpoint
from uzner.training.data_setup import load_split
from uzner.training.engine import TrainRequest, train_experiment


def test_frozen_predictor_checkpoint_and_decoder_parity(tmp_path, monkeypatch):
    """Frozen путь воспроизводит общий GP decoder и не читает dev gold."""
    path = _tiny_project(tmp_path)
    value = yaml.safe_load(path.read_text())
    value["experiment"]["model"] = {
        "architecture": "span",
        "head": "global_pointer",
        "decoder": "span",
    }
    path.write_text(yaml.safe_dump(value))
    result = train_experiment(
        TrainRequest(
            path, tmp_path, publish_summary=False, output_root_override=tmp_path / "isolated"
        )
    )
    monkeypatch.setattr(
        frozen,
        "load_model_checkpoint",
        lambda path, config, device: load_model_checkpoint(path, config, torch.device("cpu")),
    )
    move = WindowBatch.to
    monkeypatch.setattr(WindowBatch, "to", lambda self, device: move(self, torch.device("cpu")))
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: nullcontext())
    predictor = frozen.FrozenPredictor(result.paths.root)
    gold = load_split(load_experiment_config(path), tmp_path, "dev", None)
    documents = tuple(Document(d.hash, d.text) for d in gold)
    predictions = predictor.predict(documents)
    expected = tuple(
        Prediction.from_mapping(json.loads(s))
        for s in result.paths.predictions.read_text().splitlines()
    )
    assert [p.to_mapping() for p in predictions] == [p.to_mapping() for p in expected]
    assert all(not p.requires_grad for p in predictor.model.parameters())
    windows = list(predictor.windows(gold, with_labels=True))
    assert windows[0].feature.labels is not None
    assert all(e.feature.labels is None for e in predictor.windows(documents))
