"""Настоящий CUDA BF16 smoke полного train/checkpoint/resume на маленьком encoder."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from tests.test_training_pipeline import _tiny_project
from uzner.training import engine
from uzner.training.requests import TrainRequest


@pytest.mark.train
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Нужна настоящая CUDA GPU")
@pytest.mark.parametrize("head", ["softmax", "crf", "global_pointer"])
def test_cuda_bf16_train_save_reload_resume(tmp_path, monkeypatch, head):
    """Проверяет GPU-обучение разных голов и полное восстановление без MLflow."""
    config_path = _tiny_project(tmp_path)
    payload = yaml.safe_load(config_path.read_text())
    config = payload["experiment"]
    config["training"].update(require_gpu=True, bf16=True)
    config["model"].update(tag_scheme="bioes", head=head)
    if head == "crf":
        config["model"]["decoder"] = "crf"
    if head == "global_pointer":
        config["model"].update(architecture="span", decoder="span", span_head_size=8)
    config_path.write_text(yaml.safe_dump(payload))
    flags = []

    def no_tracker(*args, **kwargs):
        """Фиксирует отсутствие публикации smoke в основной MLflow."""
        flags.append(kwargs["enabled"])
        return None

    monkeypatch.setattr(engine, "start_mlflow_run", no_tracker)
    request = dict(
        config_path=config_path,
        project_root=tmp_path,
        publish_summary=False,
        output_root_override=tmp_path / "smoke",
    )
    first = engine.train_experiment(TrainRequest(**request))
    assert first.paths.best_checkpoint.joinpath("head.safetensors").is_file()
    payload["experiment"]["training"]["epochs"] = 2
    config_path.write_text(yaml.safe_dump(payload))
    last = engine.train_experiment(TrainRequest(**request, resume=True))
    state = json.loads((last.paths.last_checkpoint / "trainer_state.json").read_text())
    metadata = json.loads(last.paths.metadata.read_text())
    assert state["epoch"] == 2
    assert "cuda" in metadata["runtime"]["device"]
    assert json.loads(last.paths.status.read_text())["status"] == "complete"
    assert flags == [False, False]
    assert last.paths.predictions.is_file()


@pytest.mark.train
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Нужна настоящая CUDA GPU")
def test_actual_train_cli_on_cuda(tmp_path):
    """Запускает именно опубликованный train.py, а не только Python API обучения."""
    config = _tiny_project(tmp_path)
    payload = yaml.safe_load(config.read_text())
    payload["experiment"]["training"].update(require_gpu=True, bf16=True)
    config.write_text(yaml.safe_dump(payload))
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/train.py"),
            "--config",
            str(config),
            "--project-root",
            str(tmp_path),
            "--smoke-output",
            str(tmp_path / "cli-smoke"),
            "--max-train-documents",
            "2",
            "--max-dev-documents",
            "2",
            "--max-epochs",
            "1",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "cli-smoke/tiny_run/checkpoints/best/head.safetensors").is_file()
    assert not (tmp_path / "mlruns").exists()
