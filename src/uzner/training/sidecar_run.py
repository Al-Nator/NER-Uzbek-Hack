"""Артефакты и MLflow для экспериментов поверх фиксированного encoder-а."""

import json
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch

from uzner.data.io import sha256_file
from uzner.domain import Document, Prediction
from uzner.evaluation.slices import evaluate_detailed
from uzner.experiments.artifacts import finalize_root_manifest, write_json, write_jsonl
from uzner.experiments.mlflow_tracking import flatten_numeric
from uzner.experiments.source_snapshot import capture_source


@dataclass
class SidecarRun:
    """Изолированный запуск без копирования исходного encoder checkpoint."""

    root: Path
    source: Path
    config: dict
    smoke: bool = False

    def __enter__(self) -> "SidecarRun":
        """Фиксирует входы и создаёт MLflow run только для полноценного опыта."""
        if self.root.exists():
            raise FileExistsError(self.root)
        if self.smoke and any(
            self.root.resolve().is_relative_to(Path(p).resolve()) for p in ("runs", "artifacts")
        ):
            raise ValueError("Smoke не должен попадать в эксперименты")
        self.root.mkdir(parents=True)
        self.started = time.time()
        paths = ["experiment_config.json", "head.safetensors", "encoder/model.safetensors"]
        metadata = {
            "source": str(self.source.resolve()),
            "checkpoint_hashes": {
                p: sha256_file(self.source / "checkpoints/best" / p) for p in paths
            },
            "data_hashes": {
                s: sha256_file(Path(f"ner_uz_hackathon_participant/data/{s}.jsonl"))
                for s in ("train", "dev")
            },
            "python": platform.python_version(),
            "torch": torch.__version__,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "device": torch.cuda.get_device_name(),
        }
        write_json(self.root / "resolved_config.json", self.config)
        write_json(self.root / "metadata.json", metadata)
        capture_source(Path.cwd(), self.root / "environment")
        shutil.copyfile("uv.lock", self.root / "environment/uv.lock")
        write_json(self.root / "status.json", {"status": "running"})
        if not self.smoke:
            mlflow.set_tracking_uri("sqlite:///" + str(Path("mlruns/mlflow.db").resolve()))
            mlflow.set_experiment("uzner-fifth-series-local")
            active = mlflow.start_run(
                run_name=self.root.name,
                tags={"uzner.kind": "frozen_encoder_research", "uzner.source": self.source.name},
            )
            write_json(self.root / "mlflow.json", {"run_id": active.info.run_id})
            mlflow.log_params(self.config)
        return self

    def log(self, values: dict[str, float], step: int = 0) -> None:
        """Пишет численные метрики локально и синхронно в MLflow."""
        record = {"step": step, "elapsed_seconds": time.time() - self.started, "values": values}
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print({"step": step, "values": dict(list(values.items())[:8])}, flush=True)
        if not self.smoke:
            mlflow.log_metrics(values, step=step)

    def evaluate(
        self,
        gold: tuple[Document, ...],
        predictions: tuple[Prediction, ...],
        train: tuple[Document, ...],
        *,
        prefix: str = "dev",
        step: int = 0,
        metric_prefix: str | None = None,
    ) -> float:
        """Применяет общую exact-span метрику и полный набор срезов/ошибок."""
        report = evaluate_detailed(gold, predictions, train, {})
        folder = self.root / prefix
        write_json(folder / "metrics.json", report.to_mapping())
        write_jsonl(
            folder / "predictions.jsonl", [p.to_mapping(include_scores=True) for p in predictions]
        )
        write_jsonl(folder / "errors.jsonl", [r.to_mapping() for r in report.errors.records])
        self.log(flatten_numeric(report.to_mapping(), prefix=metric_prefix or prefix), step)
        return report.overall.micro.f1

    def __exit__(self, kind: object, error: object, traceback: object) -> None:
        """Завершает статус и публикует только лёгкие артефакты."""
        status = "complete" if kind is None else "failed"
        write_json(
            self.root / "status.json", {"status": status, "error": str(error) if error else None}
        )
        finalize_root_manifest(self.root, self.root / "artifact_manifest.json")
        if not self.smoke:
            final_metrics = self.root / "dev/metrics.json"
            if kind is None and final_metrics.is_file():
                mlflow.log_metrics(
                    flatten_numeric(json.loads(final_metrics.read_text()), prefix="best/eval"),
                    step=0,
                )
            for path in self.root.rglob("*"):
                if path.is_file() and path.suffix in {".json", ".jsonl", ".gz", ".txt", ".lock"}:
                    relative = path.parent.relative_to(self.root).as_posix()
                    mlflow.log_artifact(str(path), None if relative == "." else relative)
            mlflow.end_run(status="FINISHED" if kind is None else "FAILED")
