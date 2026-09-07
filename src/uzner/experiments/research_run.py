"""Общий журнал MLflow и лёгкие артефакты исследовательских запусков серии 7."""

import json
import os
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch

from uzner.data.io import sha256_file, write_predictions
from uzner.domain import Document, Prediction
from uzner.evaluation.slices import evaluate_detailed
from uzner.experiments.artifacts import finalize_root_manifest, write_json, write_jsonl
from uzner.experiments.mlflow_tracking import flatten_numeric
from uzner.experiments.source_snapshot import capture_source


@dataclass
class ResearchRun:
    """Создаёт отдельный run и не копирует веса в MLflow."""

    root: Path
    config: dict
    inputs: tuple[Path, ...]
    kind: str

    def __enter__(self):
        """Публикует начальное состояние до вычисления метрик."""
        if self.root.exists():
            raise FileExistsError(self.root)
        self.root.mkdir(parents=True)
        self.started = time.time()
        write_json(self.root / "resolved_config.json", self.config)
        write_json(
            self.root / "metadata.json",
            {
                "kind": self.kind,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "inputs": {str(p): sha256_file(p) for p in self.inputs},
            },
        )
        capture_source(Path.cwd(), self.root / "environment")
        if Path("uv.lock").is_file():
            shutil.copy2("uv.lock", self.root / "environment/uv.lock")
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "sqlite:///" + str(Path("mlruns/mlflow.db").resolve()))
        )
        mlflow.set_experiment(os.getenv("UZNER_MLFLOW_EXPERIMENT", "uzner-seventh-series"))
        active = mlflow.start_run(run_name=self.root.name, tags={"uzner.kind": self.kind})
        self.run_id = active.info.run_id
        write_json(self.root / "mlflow.json", {"run_id": self.run_id})
        link = self.root / "logs/mlflow_run_id.txt"
        link.parent.mkdir(exist_ok=True)
        link.write_text(self.run_id + "\n", encoding="utf-8")
        mlflow.log_params(
            {k: json.dumps(v) if isinstance(v, dict | list) else v for k, v in self.config.items()}
        )
        write_json(self.root / "status.json", {"status": "running", "kind": self.kind})
        return self

    def log(self, values: dict[str, float], step: int = 0) -> None:
        """Сохраняет шаговые метрики и печатает компактный прогресс."""
        with (self.root / "logs/events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"step": step, "elapsed": time.time() - self.started, "values": values})
                + "\n"
            )
        mlflow.log_metrics(values, step=step)
        display = (
            values
            if len(values) <= 12
            else {k: v for k, v in values.items() if k.endswith("/f1") and "overall" in k}
        )
        print({"run": self.root.name, "step": step, **display}, flush=True)

    def evaluate(
        self,
        documents: tuple[Document, ...],
        predictions: tuple[Prediction, ...],
        train: tuple[Document, ...],
        prefix: str,
    ) -> float:
        """Считает общий exact evaluator со срезами и сохраняет предсказания."""
        report = evaluate_detailed(documents, predictions, train, {})
        write_json(self.root / f"metrics/{prefix}.json", report.to_mapping())
        write_predictions(self.root / f"predictions/{prefix}.jsonl", predictions)
        write_jsonl(
            self.root / f"metrics/{prefix}_errors.jsonl",
            [r.to_mapping() for r in report.errors.records],
        )
        self.log(flatten_numeric(report.to_mapping(), prefix=prefix))
        return report.overall.micro.f1

    def __exit__(self, kind, error, traceback) -> None:
        """Фиксирует успех/ошибку и переносит только лёгкие файлы в MLflow."""
        status = "complete" if kind is None else "failed"
        write_json(
            self.root / "status.json",
            {
                "status": status,
                "kind": self.kind,
                "error": str(error) if error else None,
                "seconds": time.time() - self.started,
            },
        )
        finalize_root_manifest(self.root, self.root / "artifact_manifest.json")
        try:
            for path in self.root.rglob("*"):
                if path.is_file() and path.suffix in {
                    ".json",
                    ".jsonl",
                    ".txt",
                    ".gz",
                    ".yaml",
                    ".lock",
                }:
                    relative = str(path.parent.relative_to(self.root))
                    mlflow.log_artifact(str(path), None if relative == "." else relative)
        finally:
            mlflow.end_run(status="FINISHED" if kind is None else "FAILED")
