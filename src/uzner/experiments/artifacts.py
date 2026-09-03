"""Единая раскладка артефактов каждого эксперимента."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from uzner.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Все стандартные пути одного запуска."""

    root: Path
    best_checkpoint: Path
    last_checkpoint: Path
    predictions: Path
    metrics: Path
    slice_metrics: Path
    log: Path
    resolved_config: Path
    metadata: Path


def prepare_run_paths(
    config: ExperimentConfig,
    *,
    project_root: Path,
    resume: bool = False,
) -> RunPaths:
    """Создаёт каталог запуска и защищает существующие результаты."""
    root = (project_root / config.output_root / config.run_id).resolve()
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"Каталог запуска уже непустой: {root}")
    checkpoints = root / "checkpoints"
    predictions = root / "predictions"
    metrics = root / "metrics"
    logs = root / "logs"
    for directory in (checkpoints, predictions, metrics, logs):
        directory.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=root,
        best_checkpoint=checkpoints / "best",
        last_checkpoint=checkpoints / "last",
        predictions=predictions / "dev.jsonl",
        metrics=metrics / "dev.json",
        slice_metrics=metrics / "slices.json",
        log=logs / "train.jsonl",
        resolved_config=root / "resolved_config.yaml",
        metadata=root / "metadata.json",
    )


def write_resolved_config(path: Path, config: ExperimentConfig) -> None:
    """Записывает полностью разрешённый конфиг запуска."""
    path.write_text(
        yaml.safe_dump(config.to_mapping(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Атомарно записывает читаемый JSON-артефакт."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
