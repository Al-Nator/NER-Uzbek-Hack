"""Типизированный контракт запуска и результата обучения."""

from dataclasses import dataclass
from pathlib import Path

from uzner.experiments.artifacts import RunPaths


@dataclass(frozen=True, slots=True)
class TrainRequest:
    """Входные параметры одного запуска и безопасного smoke-режима."""

    config_path: Path
    project_root: Path
    resume: bool = False
    max_train_documents: int | None = None
    max_dev_documents: int | None = None
    max_epochs: int | None = None
    publish_summary: bool = True
    output_root_override: Path | None = None
    run_id_suffix: str | None = None


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Итог завершённого запуска."""

    paths: RunPaths
    best_epoch: int
    best_micro_f1: float
