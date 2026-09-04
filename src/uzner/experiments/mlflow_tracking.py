"""Локальный MLflow-трекинг без дублирования тяжёлых checkpoint-ов."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from uzner.config import ExperimentConfig
from uzner.evaluation.slices import DetailedEvaluation
from uzner.evaluation.tokenizer_audit import TokenizerAudit
from uzner.experiments.artifacts import ArtifactEntry, RunPaths


@dataclass(frozen=True, slots=True)
class MlflowSettings:
    """Разрешённые настройки локального или удалённого MLflow."""

    tracking_uri: str
    experiment_name: str
    artifact_location: str | None
    system_metrics: bool

    @classmethod
    def from_environment(cls, project_root: Path) -> MlflowSettings:
        """Читает env и строит воспроизводимый local-first backend."""
        root = (project_root / "mlruns").resolve()
        root.mkdir(parents=True, exist_ok=True)
        explicit_uri = os.getenv("MLFLOW_TRACKING_URI")
        tracking_uri = explicit_uri or f"sqlite:///{root / 'mlflow.db'}"
        return cls(
            tracking_uri=tracking_uri,
            experiment_name=os.getenv("UZNER_MLFLOW_EXPERIMENT", "uzner-first-series"),
            artifact_location=None if explicit_uri else (root / "artifacts").as_uri(),
            system_metrics=_environment_flag("UZNER_MLFLOW_SYSTEM_METRICS", default=True),
        )


def _environment_flag(name: str, *, default: bool) -> bool:
    """Читает обычный булев флаг из переменной окружения."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.casefold() not in {"0", "false", "no", "off"}


def _metric_key(value: str) -> str:
    """Приводит путь метрики к безопасному формату MLflow."""
    replacements = (("<=", "le_"), (">=", "ge_"), ("<", "lt_"), (">", "gt_"))
    for source, target in replacements:
        value = value.replace(source, target)
    return re.sub(r"[^A-Za-z0-9_./-]+", "_", value).strip("_/")


def flatten_numeric(value: object, *, prefix: str = "") -> dict[str, float]:
    """Разворачивает вложенный отчёт, оставляя только числовые метрики."""
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for name, nested in value.items():
            path = f"{prefix}/{name}" if prefix else str(name)
            flattened.update(flatten_numeric(nested, prefix=path))
    elif isinstance(value, int | float) and not isinstance(value, bool) and prefix:
        flattened[_metric_key(prefix)] = float(value)
    return flattened


def _flatten_params(value: object, *, prefix: str = "") -> dict[str, str]:
    """Разворачивает конфиг и metadata в строковые MLflow params."""
    flattened: dict[str, str] = {}
    if isinstance(value, dict):
        for name, nested in value.items():
            path = f"{prefix}.{name}" if prefix else str(name)
            flattened.update(_flatten_params(nested, prefix=path))
    elif prefix and value is not None:
        flattened[prefix] = str(value)
    return flattened


def _load_mlflow() -> Any:
    """Импортирует MLflow только для настоящего experiment run-а."""
    import mlflow

    return mlflow


def _ensure_active_experiment(mlflow: Any, settings: MlflowSettings) -> str:
    """Создаёт experiment или восстанавливает его после soft-delete."""
    experiment = mlflow.get_experiment_by_name(settings.experiment_name)
    if experiment is None:
        return str(
            mlflow.create_experiment(
                settings.experiment_name,
                artifact_location=settings.artifact_location,
            )
        )
    experiment_id = str(experiment.experiment_id)
    if getattr(experiment, "lifecycle_stage", "active") == "deleted":
        client = mlflow.MlflowClient(tracking_uri=settings.tracking_uri)
        client.restore_experiment(experiment_id)
    return experiment_id


@dataclass(slots=True)
class MlflowTracker:
    """Связывает один локальный run-каталог с одним MLflow run."""

    mlflow: Any
    mlflow_run_id: str

    def log_train_step(self, values: dict[str, object], *, step: int) -> None:
        """Логирует loss и runtime одного optimizer-интервала."""
        names = {
            "loss": "train/loss_step",
            "learning_rate": "optimizer/learning_rate",
            "gradient_norm": "optimizer/gradient_norm",
            "tokens_per_second": "runtime/train_tokens_per_second_step",
            "gpu_memory_gib": "runtime/gpu_memory_allocated_gib",
        }
        metrics = {
            names[name]: float(value)
            for name, value in values.items()
            if name in names and value is not None
        }
        self.mlflow.log_metrics(metrics, step=step)

    def log_reference_epoch(self, values: dict[str, object], *, epoch: int) -> None:
        """Логирует в MLflow loss каждой эпохи official baseline."""
        metrics = {
            f"epoch/{name}": float(values[name])
            for name in ("train_loss", "dev_loss")
            if values.get(name) is not None
        }
        self.mlflow.log_metrics(metrics, step=epoch)

    def log_epoch(
        self,
        record: object,
        evaluation: DetailedEvaluation,
        *,
        best: bool,
    ) -> None:
        """Логирует train/runtime и полную exact/slice диагностику эпохи."""
        values = asdict(record)  # type: ignore[arg-type]
        epoch = int(values.pop("epoch"))
        metrics = flatten_numeric(values, prefix="epoch")
        metrics.update(flatten_numeric(evaluation.to_mapping(), prefix="eval"))
        metrics["selection/is_best"] = float(best)
        if values["eval_documents_per_second"]:
            metrics["runtime/eval_milliseconds_per_document"] = (
                1000.0 / values["eval_documents_per_second"]
            )
        self.mlflow.log_metrics(metrics, step=epoch)

    def log_metadata(self, metadata: dict[str, object]) -> None:
        """Добавляет хэши данных, размеры выборок и runtime как params."""
        self.mlflow.log_params(_flatten_params(metadata, prefix="run"))

    def finish(
        self,
        paths: RunPaths,
        evaluation: DetailedEvaluation,
        audit: TokenizerAudit,
        entries: tuple[ArtifactEntry, ...],
    ) -> None:
        """Логирует финальную диагностику и лёгкие артефакты, закрывая run."""
        metrics = flatten_numeric(evaluation.to_mapping(), prefix="best/eval")
        metrics.update(flatten_numeric(audit.to_mapping(), prefix="best/tokenizer"))
        metrics.update(_artifact_metrics(entries))
        self.mlflow.log_metrics(metrics)
        self.mlflow.log_artifact(str(paths.resolved_config), artifact_path="run")
        self.mlflow.log_artifact(str(paths.metadata), artifact_path="run")
        self.mlflow.log_artifact(str(paths.status), artifact_path="run")
        self.mlflow.log_artifact(str(paths.manifest), artifact_path="run")
        for name in ("metrics", "logs", "predictions", "reports", "environment"):
            directory = paths.root / name
            if directory.exists():
                self.mlflow.log_artifacts(str(directory), artifact_path=name)
        self.mlflow.end_run(status="FINISHED")

    def fail(self, paths: RunPaths, error: Exception) -> None:
        """Помечает MLflow run ошибочным и сохраняет доступную диагностику."""
        self.mlflow.set_tags({"uzner.status": "failed", "uzner.error_type": type(error).__name__})
        if paths.status.exists():
            self.mlflow.log_artifact(str(paths.status), artifact_path="run")
        if paths.events.exists():
            self.mlflow.log_artifact(str(paths.events), artifact_path="logs")
        self.mlflow.end_run(status="FAILED")


def _artifact_metrics(entries: tuple[ArtifactEntry, ...]) -> dict[str, float]:
    """Считает объём полного run-а и checkpoint-ов без их копирования."""
    return {
        "artifacts/file_count": float(len(entries)),
        "artifacts/total_bytes": float(sum(entry.bytes for entry in entries)),
        "artifacts/best_checkpoint_bytes": float(
            sum(entry.bytes for entry in entries if entry.path.startswith("checkpoints/best/"))
        ),
        "artifacts/last_checkpoint_bytes": float(
            sum(entry.bytes for entry in entries if entry.path.startswith("checkpoints/last/"))
        ),
    }


def start_mlflow_run(
    config: ExperimentConfig,
    paths: RunPaths,
    project_root: Path,
    *,
    enabled: bool,
    resume: bool,
) -> MlflowTracker | None:
    """Открывает или восстанавливает MLflow run; smoke/test можно отключить."""
    if not enabled or not _environment_flag("UZNER_MLFLOW_ENABLED", default=True):
        return None
    mlflow = _load_mlflow()
    settings = MlflowSettings.from_environment(project_root)
    mlflow.set_tracking_uri(settings.tracking_uri)
    experiment_id = _ensure_active_experiment(mlflow, settings)
    link = paths.events.parent / "mlflow_run_id.txt"
    if resume:
        if not link.is_file():
            raise FileNotFoundError("Resume требует logs/mlflow_run_id.txt")
        active = mlflow.start_run(
            run_id=link.read_text(encoding="utf-8").strip(),
            log_system_metrics=settings.system_metrics,
        )
    else:
        active = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=config.run_id,
            tags={
                "uzner.run_id": config.run_id,
                "uzner.pipeline": config.pipeline,
                "uzner.encoder": config.encoder.name,
                "uzner.encoder_revision": config.encoder.revision,
                "uzner.tag_scheme": config.model.tag_scheme,
                "uzner.head": config.model.head,
                "uzner.decoder": config.model.decoder,
            },
            log_system_metrics=settings.system_metrics,
        )
        link.write_text(active.info.run_id + "\n", encoding="utf-8")
        mlflow.log_params(_flatten_params(config.to_mapping(), prefix="config"))
    return MlflowTracker(mlflow, active.info.run_id)
