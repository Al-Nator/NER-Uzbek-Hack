"""Короткие CPU end-to-end проверки sidecar-артефактов и обучения char-head."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from uzner.domain import Document, Entity, Prediction
from uzner.models.char_boundary import CharConfig
from uzner.models.memory_knn import KnnConfig
from uzner.training import char_pipeline, memory_pipeline, sidecar_run
from uzner.training.char_features import boundary_batch
from uzner.training.sidecar_run import SidecarRun


class TinyPredictor:
    """Имитирует фиксированный encoder с согласованным span-декодированием."""

    def __init__(self, source):
        """Принимает путь без обращения к модельным весам."""

    def windows(self, documents, *, with_labels=False):
        """Возвращает один предсказанный NAME и один фоновый токен."""
        for index, _doc in enumerate(documents):
            feature = SimpleNamespace(
                offsets=((0, 0), (0, 3), (4, 7)), labels=(-100, 8, 0), window_index=0
            )
            scores = torch.zeros(3, 3, 3)
            scores[1, 1, 1] = 0.9
            yield SimpleNamespace(
                document_index=index, feature=feature, hidden=torch.ones(3, 8), probabilities=scores
            )

    def predict(self, documents):
        """Возвращает тот же NAME, что и decoder синтетических scores."""
        return tuple(Prediction(d.hash, (Entity(0, 3, "NAME", 0.9),)) for d in documents)


def prepare_source(path: Path) -> Path:
    """Создаёт маленькие входы для проверки фиксации SHA-256."""
    source = path / "source"
    for name in ("experiment_config.json", "head.safetensors", "encoder/model.safetensors"):
        target = source / "checkpoints/best" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture", encoding="utf-8")
    return source


@pytest.mark.parametrize("kind", ["knn", "char"])
def test_cpu_frozen_end_to_end(tmp_path, monkeypatch, kind):
    """Сохраняет метрики, предсказания, provenance и checkpoint-ы без MLflow/GPU."""
    source = prepare_source(tmp_path)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU smoke")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(memory_pipeline, "FrozenPredictor", TinyPredictor)
    monkeypatch.setattr(char_pipeline, "FrozenPredictor", TinyPredictor)
    monkeypatch.setattr(
        char_pipeline,
        "boundary_batch",
        lambda examples, device: boundary_batch(examples, torch.device("cpu")),
    )
    train = (Document("t", "Ali Vali", (Entity(0, 3, "NAME"),)),)
    dev = (Document("d", "Ali Said", (Entity(0, 3, "NAME"),)),)
    config = KnnConfig(sampling_rate=1) if kind == "knn" else CharConfig(epochs=2, batch_size=1)
    root = tmp_path / "run"
    with SidecarRun(root, source, asdict(config), smoke=True) as run:
        (memory_pipeline.run_memory if kind == "knn" else char_pipeline.run_character)(
            source, train, dev, config, run
        )
    assert json.loads((root / "status.json").read_text())["status"] == "complete"
    assert (root / "environment/source_manifest.json").is_file()
    assert (root / "dev/predictions.jsonl").is_file()
    assert not (root / "mlflow.json").exists()
    if kind == "char":
        saved = torch.load(root / "checkpoints/best.pt", weights_only=False)
        assert "optimizer" in saved and "alphabet" in saved and "source" in saved
    else:
        assert (root / "memory.pt").is_file()
    with pytest.raises(FileExistsError), SidecarRun(root, source, {}, smoke=True):
        pass


def test_sidecar_marks_failure(tmp_path, monkeypatch):
    """Сбой не превращается в завершённый исследовательский run."""
    source = prepare_source(tmp_path)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU smoke")
    root = tmp_path / "failed"
    with pytest.raises(RuntimeError), SidecarRun(root, source, {}, smoke=True):
        raise RuntimeError("test")
    assert json.loads((root / "status.json").read_text())["status"] == "failed"


def test_sidecar_tracking_without_real_mlflow(tmp_path, monkeypatch):
    """Публикует метрики/лёгкие артефакты, но не память и не checkpoint."""
    source = prepare_source(tmp_path)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU smoke")
    artifacts, ended, metrics = [], [], []
    fake = SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        set_experiment=lambda name: None,
        start_run=lambda **kw: SimpleNamespace(info=SimpleNamespace(run_id="fake-only")),
        log_params=lambda params: None,
        log_metrics=lambda values, step: metrics.append(values),
        log_artifact=lambda path, folder: artifacts.append((path, folder)),
        end_run=lambda status: ended.append(status),
    )
    monkeypatch.setattr(sidecar_run, "mlflow", fake)
    root = tmp_path / "tracked"
    with SidecarRun(root, source, {"seed": 42}) as run:
        run.log({"train/loss": 1.0})
        (root / "memory.pt").write_bytes(b"test")
    assert ended == ["FINISHED"] and metrics
    assert artifacts and not any(p.endswith("memory.pt") for p, _ in artifacts)
    assert all(folder != "." for _, folder in artifacts)
    assert any(folder is None for _, folder in artifacts)
    assert (root / "artifact_manifest.json").is_file()
