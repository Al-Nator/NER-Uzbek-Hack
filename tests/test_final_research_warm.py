"""Явное продолжение research-checkpoint на train+dev без validation."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from tests.test_training_pipeline import _tiny_project
from uzner.config import load_experiment_config
from uzner.training.final_fit import FinalFitRequest, run_final_fit
from uzner.training.final_support import initialize_final


def test_research_warm_start_guards_and_no_validation(tmp_path, monkeypatch):
    """Опция явная, данные проверяются по hash, optimizer новый, dev-F1 отсутствует."""
    path = _tiny_project(tmp_path, epochs=1)
    payload = yaml.safe_load(path.read_text())
    exp = payload["experiment"]
    exp["training"]["early_stopping_patience"] = 0
    path.write_text(yaml.safe_dump(payload))
    monkeypatch.setattr("uzner.training.final_fit.start_mlflow_run", lambda *a, **k: None)
    first = run_final_fit(FinalFitRequest(path, tmp_path, tmp_path / "smoke"))
    status = first / "status.json"
    status.write_text(json.dumps({"status": "complete", "best_micro_f1": 0.9}))
    metadata_path = first / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    for item in metadata["data"].values():
        item["path"] = "/previous-host/data.jsonl"
    metadata_path.write_text(json.dumps(metadata))
    exp["run_id"] = "research_warm"
    exp["training"].update(
        learning_rate=1e-6,
        initial_checkpoint=str(first / "checkpoints/best"),
    )
    path.write_text(yaml.safe_dump(payload))
    config = load_experiment_config(path)
    with pytest.raises(ValueError, match="final-fit"):
        initialize_final(config, tmp_path, torch.device("cpu"))
    second = run_final_fit(
        FinalFitRequest(path, tmp_path, tmp_path / "smoke", allow_research_initial=True)
    )
    result = json.loads((second / "metadata.json").read_text())
    assert result["allow_research_initial"] is True
    assert result["initial_checkpoint"]["optimizer_state"] == "reset"
    assert result["initial_checkpoint"]["scheduler_state"] == "reset"
    assert result["selected_epoch"] == 1
    assert not (second / "metrics/dev.json").exists()
    status.write_text(json.dumps({"status": "running", "best_micro_f1": 0.9}))
    with pytest.raises(ValueError, match="завершённый"):
        initialize_final(config, tmp_path, torch.device("cpu"), allow_research_initial=True)
    status.write_text(json.dumps({"status": "complete", "best_micro_f1": 0.9}))
    next(iter(metadata["data"].values()))["sha256"] = "invalid"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="исходные train/dev"):
        initialize_final(config, tmp_path, torch.device("cpu"), allow_research_initial=True)


def test_f15_recipe_preserves_s33_architecture():
    """Меняется только короткий training budget и источник уже готовых весов."""
    target = load_experiment_config(Path("configs/final/f15_s33_train_dev_warm_epoch1.yaml"))
    reference = load_experiment_config(Path("configs/final/f11_s33_train_dev_low_lr_epoch1.yaml"))
    assert (target.encoder, target.model, target.tokenization) == (
        reference.encoder, reference.model, reference.tokenization
    )
    assert target.training == replace(
        reference.training,
        epochs=1,
        learning_rate=1e-6,
        initial_checkpoint="runs/s33_bge_global_pointer_low_lr_a100-low-lr-v1/checkpoints/best",
    )
