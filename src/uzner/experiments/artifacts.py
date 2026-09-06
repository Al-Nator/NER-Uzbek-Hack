"""Единая раскладка артефактов каждого эксперимента."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from uzner.config import ExperimentConfig
from uzner.data.io import sha256_file


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Все стандартные пути одного запуска."""

    root: Path
    best_checkpoint: Path
    last_checkpoint: Path
    predictions: Path
    metrics: Path
    slice_metrics: Path
    errors: Path
    error_summary: Path
    tokenizer_audit: Path
    events: Path
    history: Path
    console_log: Path
    report: Path
    training_plot: Path
    resolved_config: Path
    metadata: Path
    status: Path
    manifest: Path
    environment: Path


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    """Один файл в итоговом манифесте запуска."""

    path: str
    bytes: int
    sha256: str


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
    reports = root / "reports"
    environment = root / "environment"
    for directory in (checkpoints, predictions, metrics, logs, reports, environment):
        directory.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=root,
        best_checkpoint=checkpoints / "best",
        last_checkpoint=checkpoints / "last",
        predictions=predictions / "dev.jsonl",
        metrics=metrics / "dev.json",
        slice_metrics=metrics / "slices.json",
        errors=metrics / "errors.jsonl",
        error_summary=metrics / "error_summary.json",
        tokenizer_audit=metrics / "tokenizer_audit.json",
        events=logs / "events.jsonl",
        history=logs / "history.csv",
        console_log=logs / "console.log",
        report=reports / "report.md",
        training_plot=reports / "training.svg",
        resolved_config=root / "resolved_config.yaml",
        metadata=root / "metadata.json",
        status=root / "status.json",
        manifest=root / "artifact_manifest.json",
        environment=environment,
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


def write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    """Атомарно записывает набор JSON-объектов по одному на строку."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def finalize_artifact_manifest(paths: RunPaths) -> tuple[ArtifactEntry, ...]:
    """Хэширует все артефакты и атомарно пишет итоговый манифест."""
    return finalize_root_manifest(paths.root, paths.manifest)


def finalize_root_manifest(root: Path, manifest: Path) -> tuple[ArtifactEntry, ...]:
    """Хэширует каталог независимо от схемы model/data/frozen-run."""
    entries = tuple(
        ArtifactEntry(
            path=str(path.relative_to(root)),
            bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest and not path.name.endswith(".tmp")
    )
    write_json(
        manifest,
        {
            "schema_version": 1,
            "run_root": str(root),
            "artifacts": [asdict(entry) for entry in entries],
        },
    )
    return entries
