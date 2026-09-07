"""Перенос завершённого MLflow run между tracking backend-ами."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

TRANSFER_SOURCE_TAG = "uzner.transfer.source_run_id"
TRANSFER_HOST_TAG = "uzner.transfer.source_host"
TRANSFER_STATUS_TAG = "uzner.transfer.status"
_BATCH_SIZE = 900


@dataclass(frozen=True, slots=True)
class MlflowTransferRequest:
    """Описывает один перенос и локальные лёгкие артефакты."""

    source_uri: str
    source_run_id: str
    source_host: str
    destination_uri: str
    destination_experiment: str
    destination_artifact_location: str
    run_root: Path


@dataclass(frozen=True, slots=True)
class MlflowTransferResult:
    """Возвращает локальный MLflow ID и признак нового импорта."""

    destination_run_id: str
    imported: bool
    metric_points: int


def _ensure_experiment(client: MlflowClient, request: MlflowTransferRequest) -> str:
    """Находит или создаёт целевой experiment с локальными артефактами."""
    experiment = client.get_experiment_by_name(request.destination_experiment)
    if experiment is None:
        return client.create_experiment(
            request.destination_experiment,
            artifact_location=request.destination_artifact_location,
        )
    if experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
    return str(experiment.experiment_id)


def _existing_transfer(
    client: MlflowClient,
    experiment_id: str,
    source_run_id: str,
) -> str | None:
    """Ищет уже завершённую копию удалённого run."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.`{TRANSFER_SOURCE_TAG}` = '{source_run_id}'",
        max_results=10,
    )
    for run in runs:
        if run.data.tags.get(TRANSFER_STATUS_TAG) == "complete":
            return str(run.info.run_id)
    return None


def _safe_tags(tags: dict[str, str], request: MlflowTransferRequest) -> dict[str, str]:
    """Копирует пользовательские теги без служебных MLflow-полей."""
    copied = {key: value for key, value in tags.items() if not key.startswith("mlflow.")}
    copied.update(
        {
            TRANSFER_SOURCE_TAG: request.source_run_id,
            TRANSFER_HOST_TAG: request.source_host,
            TRANSFER_STATUS_TAG: "copying",
        }
    )
    return copied


def _chunks(values: list[Any], size: int = _BATCH_SIZE) -> list[list[Any]]:
    """Разбивает большой пакет на допустимые MLflow-батчи."""
    return [values[index : index + size] for index in range(0, len(values), size)]


def _copy_metrics(
    source: MlflowClient,
    destination: MlflowClient,
    source_run: Any,
    destination_run_id: str,
) -> int:
    """Копирует полную историю каждой метрики."""
    points = 0
    for key in sorted(source_run.data.metrics):
        history = list(source.get_metric_history(source_run.info.run_id, key))
        for batch in _chunks(history):
            destination.log_batch(destination_run_id, metrics=batch)
        points += len(history)
    return points


def _log_light_artifacts(client: MlflowClient, run_id: str, root: Path) -> None:
    """Загружает в MLflow только лёгкие run-артефакты."""
    for name in (
        "resolved_config.yaml",
        "resolved_config.json",
        "metadata.json",
        "status.json",
        "artifact_manifest.json",
        "holdout_protocol.json",
    ):
        path = root / name
        if path.is_file():
            client.log_artifact(run_id, str(path), artifact_path="run")
    for name in ("metrics", "logs", "predictions", "reports", "environment"):
        path = root / name
        if path.is_dir():
            client.log_artifacts(run_id, str(path), artifact_path=name)


def transfer_mlflow_run(request: MlflowTransferRequest) -> MlflowTransferResult:
    """Идемпотентно переносит run в основной локальный MLflow."""
    from mlflow.entities import Param
    from mlflow.tracking import MlflowClient

    if not request.run_root.is_dir():
        raise FileNotFoundError(f"Не найден run-каталог: {request.run_root}")
    source = MlflowClient(tracking_uri=request.source_uri)
    destination = MlflowClient(tracking_uri=request.destination_uri)
    source_run = source.get_run(request.source_run_id)
    for _attempt in range(30):
        if source_run.info.status != "RUNNING":
            break
        sleep(1)
        source_run = source.get_run(request.source_run_id)
    if source_run.info.status == "RUNNING":
        raise RuntimeError("Удалённый MLflow run ещё не завершён")
    experiment_id = _ensure_experiment(destination, request)
    existing = _existing_transfer(destination, experiment_id, request.source_run_id)
    if existing is not None:
        return MlflowTransferResult(existing, False, 0)

    run_name = source_run.data.tags.get("mlflow.runName")
    created = destination.create_run(
        experiment_id,
        start_time=source_run.info.start_time,
        tags=_safe_tags(source_run.data.tags, request),
        run_name=run_name,
    )
    destination_run_id = str(created.info.run_id)
    try:
        params = [Param(key, value) for key, value in sorted(source_run.data.params.items())]
        for batch in _chunks(params):
            destination.log_batch(destination_run_id, params=batch)
        metric_points = _copy_metrics(source, destination, source_run, destination_run_id)
        _log_light_artifacts(destination, destination_run_id, request.run_root)
        destination.set_tag(destination_run_id, TRANSFER_STATUS_TAG, "complete")
        destination.set_terminated(
            destination_run_id,
            status=source_run.info.status,
            end_time=source_run.info.end_time,
        )
    except Exception:
        destination.set_tag(destination_run_id, TRANSFER_STATUS_TAG, "failed")
        destination.set_terminated(destination_run_id, status="FAILED")
        raise
    return MlflowTransferResult(destination_run_id, True, metric_points)
