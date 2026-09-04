"""Тесты полного MLflow-контракта без запуска внешнего сервера."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.helpers import sample_documents
from uzner.config import load_experiment_config, with_run_suffix
from uzner.domain import Prediction
from uzner.evaluation.slices import evaluate_detailed
from uzner.evaluation.tokenizer_audit import AuditSlice, TokenizerAudit
from uzner.experiments.artifacts import (
    finalize_artifact_manifest,
    prepare_run_paths,
    write_json,
    write_resolved_config,
)
from uzner.experiments.logging import EpochRecord, RunLogger
from uzner.experiments.mlflow_tracking import (
    MlflowSettings,
    flatten_numeric,
    start_mlflow_run,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/e10_xlmr_base_bio.yaml"


class FakeMlflow:
    """Запоминает вызовы публичного MLflow API в памяти."""

    def __init__(self) -> None:
        """Создаёт пустые коллекции вызовов."""
        self.tracking_uri = ""
        self.params: dict[str, str] = {}
        self.tags: dict[str, str] = {}
        self.metrics: list[tuple[dict[str, float], int | None]] = []
        self.artifacts: list[tuple[str, str]] = []
        self.started: list[dict[str, object]] = []
        self.ended: list[str] = []
        self.restored: list[str] = []
        self.experiment = None

    def set_tracking_uri(self, value: str) -> None:
        """Запоминает backend URI."""
        self.tracking_uri = value

    def get_experiment_by_name(self, _name: str) -> object | None:
        """Возвращает уже созданный эксперимент."""
        return self.experiment

    def create_experiment(self, _name: str, *, artifact_location: str | None) -> str:
        """Создаёт фиктивный эксперимент."""
        self.experiment = SimpleNamespace(experiment_id="experiment-1", lifecycle_stage="active")
        self.artifacts.append((artifact_location or "", "experiment"))
        return "experiment-1"

    def MlflowClient(self, *, tracking_uri: str) -> FakeMlflow:  # noqa: N802
        """Возвращает фиктивный client для lifecycle-операций."""
        assert tracking_uri == self.tracking_uri
        return self

    def restore_experiment(self, experiment_id: str) -> None:
        """Запоминает восстановление soft-deleted experiment."""
        self.restored.append(experiment_id)
        self.experiment.lifecycle_stage = "active"

    def start_run(self, **kwargs: object) -> object:
        """Открывает фиктивный run с постоянным ID."""
        self.started.append(kwargs)
        return SimpleNamespace(info=SimpleNamespace(run_id="mlflow-run-1"))

    def log_params(self, values: dict[str, str]) -> None:
        """Сохраняет параметры."""
        self.params.update(values)

    def log_metrics(self, values: dict[str, float], *, step: int | None = None) -> None:
        """Сохраняет пакет метрик и его step."""
        self.metrics.append((values, step))

    def set_tags(self, values: dict[str, str]) -> None:
        """Сохраняет теги ошибки."""
        self.tags.update(values)

    def log_artifact(self, path: str, *, artifact_path: str) -> None:
        """Сохраняет путь одиночного артефакта."""
        self.artifacts.append((path, artifact_path))

    def log_artifacts(self, path: str, *, artifact_path: str) -> None:
        """Сохраняет путь каталога артефактов."""
        self.artifacts.append((path, artifact_path))

    def end_run(self, *, status: str) -> None:
        """Запоминает финальный статус."""
        self.ended.append(status)


def _record() -> EpochRecord:
    """Создаёт строку эпохи с runtime-метриками."""
    return EpochRecord(
        1, 0.5, 0.4, 0.8, 0.75, 0.77, 0.76, 0.7, 0.8, 0.78, 2e-5, 0.9, 2, 1, 100, 20, 1.5
    )


def _evaluation() -> object:
    """Строит идеальную detailed evaluation со всеми срезами."""
    documents = sample_documents()
    predictions = tuple(Prediction(item.hash, item.entities) for item in documents)
    return evaluate_detailed(documents, predictions, documents, {})


def _audit() -> TokenizerAudit:
    """Строит маленький tokenizer audit."""
    overall = AuditSlice(4, 4, 1.0, 1.25, 2.0, 4.0)
    return TokenizerAudit("fake", "rev", 512, 128, 2, 2, 1.0, 1, overall, {})


def test_mlflow_logs_steps_slices_metadata_and_light_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tracker сохраняет подробные ряды и не копирует checkpoint-каталоги."""
    config = load_experiment_config(CONFIG_PATH)
    paths = prepare_run_paths(config, project_root=tmp_path)
    fake = FakeMlflow()
    monkeypatch.setattr("uzner.experiments.mlflow_tracking._load_mlflow", lambda: fake)
    tracker = start_mlflow_run(config, paths, tmp_path, enabled=True, resume=False)
    assert tracker is not None
    logger = RunLogger(config.run_id, paths, tracker)
    evaluation = _evaluation()

    write_resolved_config(paths.resolved_config, config)
    write_json(paths.metadata, {"data": {"dev": {"sha256": "abc"}}, "documents": 2})
    write_json(paths.status, {"status": "complete"})
    logger.train_step(
        epoch=1,
        step=25,
        loss=0.4,
        learning_rate=2e-5,
        gradient_norm=0.9,
        tokens_per_second=100,
        gpu_memory_gib=1.5,
    )
    tracker.log_reference_epoch({"train_loss": 0.8, "dev_loss": 0.5}, epoch=1)
    logger.epoch(_record(), evaluation, best=True)
    tracker.log_metadata({"data": {"dev": {"sha256": "abc"}}, "documents": 2})
    paths.predictions.write_text("{}\n", encoding="utf-8")
    paths.report.write_text("report\n", encoding="utf-8")
    entries = finalize_artifact_manifest(paths)
    tracker.finish(paths, evaluation, _audit(), entries)

    all_metrics = {name for batch, _step in fake.metrics for name in batch}
    assert fake.tracking_uri.startswith("sqlite:////")
    assert fake.params["config.encoder.name"] == config.encoder.name
    assert fake.params["run.data.dev.sha256"] == "abc"
    assert "train/loss_step" in all_metrics
    assert "epoch/train_loss" in all_metrics
    assert "epoch/dev_loss" in all_metrics
    assert "eval/overall/by_label/ORG/f1" in all_metrics
    assert "best/tokenizer/overall/representability" in all_metrics
    assert "artifacts/best_checkpoint_bytes" in all_metrics
    assert not any(target == "checkpoints" for _path, target in fake.artifacts)
    assert fake.ended == ["FINISHED"]


def test_mlflow_resume_failure_flags_and_safe_metric_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume использует прежний ID, а ошибки получают FAILED и артефакт status."""
    config = load_experiment_config(CONFIG_PATH)
    paths = prepare_run_paths(config, project_root=tmp_path)
    fake = FakeMlflow()
    fake.experiment = SimpleNamespace(experiment_id="experiment-1")
    monkeypatch.setattr("uzner.experiments.mlflow_tracking._load_mlflow", lambda: fake)
    link = paths.events.parent / "mlflow_run_id.txt"
    link.write_text("existing-run\n", encoding="utf-8")
    tracker = start_mlflow_run(config, paths, tmp_path, enabled=True, resume=True)
    assert tracker is not None
    write_json(paths.status, {"status": "failed"})
    paths.events.write_text("event\n", encoding="utf-8")
    tracker.fail(paths, ValueError("bad"))

    assert fake.started[0]["run_id"] == "existing-run"
    assert fake.tags["uzner.error_type"] == "ValueError"
    assert fake.ended == ["FAILED"]
    assert flatten_numeric({"slice": {"<=4": 0.5}}, prefix="eval") == {"eval/slice/le_4": 0.5}


def test_mlflow_restores_soft_deleted_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Soft-deleted experiment восстанавливается перед новым run-ом."""
    config = load_experiment_config(CONFIG_PATH)
    paths = prepare_run_paths(config, project_root=tmp_path)
    fake = FakeMlflow()
    fake.experiment = SimpleNamespace(experiment_id="deleted-experiment", lifecycle_stage="deleted")
    monkeypatch.setattr("uzner.experiments.mlflow_tracking._load_mlflow", lambda: fake)

    tracker = start_mlflow_run(config, paths, tmp_path, enabled=True, resume=False)

    assert tracker is not None
    assert fake.restored == ["deleted-experiment"]
    assert fake.started[0]["experiment_id"] == "deleted-experiment"


def test_mlflow_can_be_disabled_and_remote_uri_has_no_local_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke отключает tracker, а внешний URI не навязывает локальный artifact store."""
    config = load_experiment_config(CONFIG_PATH)
    paths = prepare_run_paths(config, project_root=tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    monkeypatch.setenv("UZNER_MLFLOW_SYSTEM_METRICS", "false")

    settings = MlflowSettings.from_environment(tmp_path)

    assert start_mlflow_run(config, paths, tmp_path, enabled=False, resume=False) is None
    assert settings.tracking_uri == "http://127.0.0.1:5000"
    assert settings.artifact_location is None
    assert settings.system_metrics is False
    with pytest.raises(FileNotFoundError, match="mlflow_run_id"):
        fake = FakeMlflow()
        monkeypatch.setattr("uzner.experiments.mlflow_tracking._load_mlflow", lambda: fake)
        start_mlflow_run(config, paths, tmp_path, enabled=True, resume=True)


def test_real_mlflow_sqlite_run_stays_inside_pytest_tmp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Настоящий MLflow создаёт SQLite run только во временном каталоге теста."""
    import mlflow

    config = load_experiment_config(CONFIG_PATH)
    paths = prepare_run_paths(config, project_root=tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("UZNER_MLFLOW_SYSTEM_METRICS", "false")
    tracker = start_mlflow_run(config, paths, tmp_path, enabled=True, resume=False)
    assert tracker is not None
    tracker.log_train_step(
        {
            "loss": 0.5,
            "learning_rate": 2e-5,
            "gradient_norm": 0.8,
            "tokens_per_second": 100.0,
            "gpu_memory_gib": 1.0,
        },
        step=1,
    )
    tracker.log_epoch(_record(), _evaluation(), best=True)
    write_resolved_config(paths.resolved_config, config)
    write_json(paths.metadata, {"documents": 2})
    write_json(paths.status, {"status": "complete"})
    paths.events.write_text("event\n", encoding="utf-8")
    paths.predictions.write_text("{}\n", encoding="utf-8")
    paths.report.write_text("report\n", encoding="utf-8")
    entries = finalize_artifact_manifest(paths)
    tracker.finish(paths, _evaluation(), _audit(), entries)

    settings = MlflowSettings.from_environment(tmp_path)
    client = mlflow.MlflowClient(tracking_uri=settings.tracking_uri)
    run = client.get_run(tracker.mlflow_run_id)
    assert run.info.status == "FINISHED"
    assert (tmp_path / "mlruns/mlflow.db").is_file()

    experiment = client.get_experiment_by_name(settings.experiment_name)
    assert experiment is not None
    client.delete_experiment(experiment.experiment_id)
    restored_config = with_run_suffix(config, "restored")
    restored_paths = prepare_run_paths(restored_config, project_root=tmp_path)
    restored = start_mlflow_run(
        restored_config,
        restored_paths,
        tmp_path,
        enabled=True,
        resume=False,
    )
    assert restored is not None
    assert client.get_experiment(experiment.experiment_id).lifecycle_stage == "active"
    restored.mlflow.end_run(status="KILLED")
