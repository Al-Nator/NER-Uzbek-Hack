"""Фиксированная финальная эпоха без утечки train-метрик в validation-рейтинг."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from tests.test_training_pipeline import _tiny_project
from uzner.config import load_experiment_config
from uzner.domain import Document
from uzner.experiments.artifacts import write_json
from uzner.experiments.remote_sync import _publish_local_result, verify_run_manifest
from uzner.training.final_fit import FinalFitRequest, final_documents, run_final_fit


def test_five_epochs_without_validation(tmp_path, monkeypatch):
    """Пять настоящих tiny-эпох дают финальный checkpoint без dev-F1 и MLflow."""
    path = _tiny_project(tmp_path, epochs=5)
    payload = yaml.safe_load(path.read_text())
    exp = payload["experiment"]
    exp["model"] = {"architecture": "span", "head": "global_pointer", "decoder": "span"}
    exp["training"]["early_stopping_patience"] = 0
    path.write_text(yaml.safe_dump(payload))
    monkeypatch.setattr("uzner.training.final_fit.start_mlflow_run", lambda *a, **k: None)
    output = run_final_fit(FinalFitRequest(path, tmp_path, tmp_path / "smoke"))
    status = json.loads((output / "status.json").read_text())
    assert status["selected_epoch"] == 5 and status["kind"] == "final_fit"
    assert "best_micro_f1" not in status
    assert not (output / "metrics/dev.json").exists()
    assert len(list((output / "metrics").glob("train_epoch_*.json"))) == 5
    assert json.loads((output / "metadata.json").read_text())["train_documents"] == 4
    for checkpoint in ("best", "last"):
        state = json.loads((output / f"checkpoints/{checkpoint}/trainer_state.json").read_text())
        assert state["epoch"] == state["best_epoch"] == 5
    verify_run_manifest(output)
    with pytest.raises(FileExistsError):
        run_final_fit(FinalFitRequest(path, tmp_path, tmp_path / "smoke"))


def test_final_fit_guards(tmp_path):
    """Запрещает отбор по validation и скрытый smoke в production."""
    path = _tiny_project(tmp_path)
    with pytest.raises(ValueError):
        run_final_fit(FinalFitRequest(path, tmp_path))


def test_data_contract(monkeypatch):
    """Рецепт s45 неизменен, кроме отключённого early stopping; dev включён в train."""
    config = load_experiment_config(Path("configs/final/f01_s45_train_dev_silver.yaml"))
    source = load_experiment_config(
        Path("configs/experiments/a100/s45_bge_gp_external_silver.yaml")
    )
    assert config.training == replace(source.training, early_stopping_patience=0)
    assert config.model == source.model and config.encoder == source.encoder
    splits = {"train": (Document("train", "Ali"),), "dev": (Document("dev", "Vali"),)}
    monkeypatch.setattr(
        "uzner.training.final_fit.load_split", lambda config, root, split, limit: splits[split]
    )
    assert final_documents(config, Path.cwd()) == splits["train"] + splits["dev"]


@pytest.mark.parametrize(
    "kind", ["final_fit", "reranker_proposer", "span_reranker", "inference_ablation"]
)
def test_final_import_does_not_publish_f1(tmp_path, monkeypatch, kind):
    """Импорт final fit не требует best-F1 и не обновляет сравнительную CSV."""
    run = tmp_path / "final"
    (run / "logs").mkdir(parents=True)
    write_json(run / "status.json", {"kind": kind, "status": "complete"})
    client = MagicMock()
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda **kwargs: client)
    _publish_local_result(MagicMock(local_root=tmp_path), run, "local-id", "unused")
    assert (run / "logs/mlflow_local_run_id.txt").read_text().strip() == "local-id"
    assert not (tmp_path / "reports/experiments.csv").exists()
