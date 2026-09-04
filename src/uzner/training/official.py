"""Адаптер неизменённого official baseline к единым артефактам."""

from __future__ import annotations

import io
import json
import sys
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import TextIO

from transformers import AutoTokenizer

from uzner.config import (
    ExperimentConfig,
    load_data_config,
    load_experiment_config,
    resolve_sources,
    with_run_suffix,
)
from uzner.data.io import load_documents, read_jsonl
from uzner.data.windows import build_window_features, collect_chunk_edges
from uzner.domain import Document, Prediction
from uzner.evaluation.slices import evaluate_detailed
from uzner.evaluation.tokenizer_audit import audit_tokenizer
from uzner.experiments.artifacts import (
    finalize_artifact_manifest,
    prepare_run_paths,
    write_json,
    write_resolved_config,
)
from uzner.experiments.logging import EpochRecord, RunLogger
from uzner.experiments.mlflow_tracking import MlflowTracker, start_mlflow_run
from uzner.training.data_setup import collect_data_hashes
from uzner.training.engine import TrainRequest, TrainResult
from uzner.training.finalization import finalize_outputs
from uzner.training.inference import InferenceResult
from uzner.training.official_kit import download_official_snapshot, load_official_modules
from uzner.training.runtime import (
    capture_environment,
    collect_runtime_info,
    resolve_device,
    set_reproducible_seed,
)


@dataclass(slots=True)
class _TeeTextIO:
    """Дублирует official-вывод в терминал и буфер артефакта."""

    terminal: TextIO
    capture: io.StringIO

    def write(self, value: str) -> int:
        """Пишет фрагмент в оба потока без задержки в терминале."""
        written = self.terminal.write(value)
        self.terminal.flush()
        self.capture.write(value)
        return written

    def flush(self) -> None:
        """Сбрасывает оба выходных потока."""
        self.terminal.flush()
        self.capture.flush()

    def isatty(self) -> bool:
        """Сохраняет TTY-поведение progress bar исходного потока."""
        return self.terminal.isatty()


def _single_source(
    config: ExperimentConfig,
    project_root: Path,
    split: str,
) -> Path:
    """Возвращает единственный файл split, который понимает official kit."""
    data = load_data_config((project_root / config.data_config).resolve())
    sources = resolve_sources(data, split=split, project_root=project_root)  # type: ignore[arg-type]
    if len(sources) != 1:
        raise ValueError("Official baseline поддерживает ровно один файл на split")
    return sources[0][1]


def _load_predictions(path: Path) -> tuple[Prediction, ...]:
    """Читает official JSONL в общий доменный тип."""
    return tuple(Prediction.from_mapping(item) for item in read_jsonl(path))


def _edges(
    documents: tuple[Document, ...],
    tokenizer: object,
    config: ExperimentConfig,
) -> dict[str, tuple[int, ...]]:
    """Собирает character-границы окон official baseline."""
    features = build_window_features(
        documents,
        tokenizer,
        config.tokenization,
        "bio",
        with_labels=False,
    )
    by_index = collect_chunk_edges(features)
    return {document.hash: by_index.get(index, ()) for index, document in enumerate(documents)}


def _official_args(
    config: ExperimentConfig,
    train_path: Path,
    dev_path: Path,
    model_path: Path,
    output_path: Path,
    request: TrainRequest,
) -> Namespace:
    """Собирает Namespace без изменения official CLI."""
    return Namespace(
        train=train_path,
        dev=dev_path,
        output_dir=output_path,
        model_name=str(model_path),
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        max_length=config.tokenization.max_length,
        stride=config.tokenization.stride,
        seed=config.training.seed,
        device="cuda" if config.training.require_gpu else "auto",
        num_workers=config.training.num_workers,
        max_train_records=request.max_train_documents,
        max_dev_records=request.max_dev_documents,
        overwrite_output_dir=False,
    )


def _epoch_record(
    summary: dict[str, object],
    inference: InferenceResult,
) -> EpochRecord:
    """Собирает единую строку лучшей official-эпохи."""
    history = summary.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("Official summary не содержит history")
    best = min(history, key=lambda item: float(item["dev_loss"]))
    overall = inference.evaluation.overall
    return EpochRecord(
        epoch=int(best["epoch"]),
        train_loss=float(best["train_loss"]),
        dev_loss=float(best["dev_loss"]),
        micro_precision=overall.micro.precision,
        micro_recall=overall.micro.recall,
        micro_f1=overall.micro.f1,
        macro_f1=overall.macro_f1,
        org_f1=overall.by_label["ORG"].f1,
        name_f1=overall.by_label["NAME"].f1,
        geo_f1=overall.by_label["GEO"].f1,
        learning_rate=0.0,
        mean_gradient_norm=0.0,
        train_seconds=0.0,
        eval_seconds=inference.seconds,
        train_tokens_per_second=0.0,
        eval_documents_per_second=inference.documents_per_second,
        peak_gpu_memory_gib=0.0,
    )


def run_official_reference(request: TrainRequest) -> TrainResult:
    """Запускает official train/predict и добавляет единые отчёты."""
    project_root = request.project_root.resolve()
    config = with_run_suffix(
        load_experiment_config(request.config_path.resolve()), request.run_id_suffix
    )
    if config.pipeline != "official_reference":
        raise ValueError("Ожидается pipeline=official_reference")
    limited = request.max_train_documents is not None or request.max_dev_documents is not None
    if limited and request.output_root_override is None:
        raise ValueError("Smoke-limit требует отдельный output_root_override")
    if request.output_root_override is not None:
        if request.publish_summary:
            raise ValueError("Smoke/output override нельзя публиковать в сводку")
        config = replace(config, output_root=str(request.output_root_override.resolve()))
    paths = prepare_run_paths(config, project_root=project_root, resume=request.resume)
    if request.resume:
        raise ValueError("Official reference не поддерживает resume")
    logger = RunLogger(config.run_id, paths)
    tracker: MlflowTracker | None = None
    write_resolved_config(paths.resolved_config, config)
    write_json(paths.status, {"status": "running"})
    captured = io.StringIO()
    try:
        tracker = start_mlflow_run(
            config,
            paths,
            project_root,
            enabled=not limited,
            resume=False,
        )
        logger.attach_tracker(tracker)
        official_predict, official_train = load_official_modules(project_root)
        device = resolve_device(config.training.require_gpu)
        set_reproducible_seed(config.training.seed)
        train_path = _single_source(config, project_root, "train")
        dev_path = _single_source(config, project_root, "dev")
        snapshot = download_official_snapshot(config.encoder)
        work = paths.root / "official_work"
        arguments = _official_args(config, train_path, dev_path, snapshot, work, request)
        started = perf_counter()
        with (
            redirect_stdout(_TeeTextIO(sys.stdout, captured)),
            redirect_stderr(_TeeTextIO(sys.stderr, captured)),
        ):
            model_dir = official_train.run(arguments)
        train_seconds = perf_counter() - started
        model_dir.rename(paths.best_checkpoint)
        summary_path = work / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_path.rename(paths.events.parent / "official_training_summary.json")
        work.rmdir()

        prediction_args = Namespace(
            model_dir=paths.best_checkpoint,
            input=dev_path,
            output=paths.predictions,
            batch_size=config.training.eval_batch_size,
            max_length=config.tokenization.max_length,
            stride=config.tokenization.stride,
            max_records=request.max_dev_documents,
            device="cuda" if device.type == "cuda" else "cpu",
        )
        predict_started = perf_counter()
        with (
            redirect_stdout(_TeeTextIO(sys.stdout, captured)),
            redirect_stderr(_TeeTextIO(sys.stderr, captured)),
        ):
            official_predict.run(prediction_args)
        predict_seconds = perf_counter() - predict_started

        train = tuple(load_documents((("official_train", train_path),)))
        dev = tuple(load_documents((("official_dev", dev_path),)))
        if request.max_train_documents is not None:
            train = train[: request.max_train_documents]
        if request.max_dev_documents is not None:
            dev = dev[: request.max_dev_documents]
        predictions = _load_predictions(paths.predictions)
        tokenizer = AutoTokenizer.from_pretrained(paths.best_checkpoint, local_files_only=True)
        evaluation = evaluate_detailed(dev, predictions, train, _edges(dev, tokenizer, config))
        inference = InferenceResult(
            predictions=predictions,
            evaluation=evaluation,
            loss=float(summary["best_dev_loss"]),
            seconds=predict_seconds,
            documents_per_second=len(dev) / predict_seconds,
        )
        audit = audit_tokenizer(dev, tokenizer, config.encoder, config.tokenization)
        record = _epoch_record(summary, inference)
        runtime = collect_runtime_info(project_root, device)
        capture_environment(paths, project_root, runtime)
        metadata = {
            "schema_version": 1,
            "run_id": config.run_id,
            "selection_metric": "dev_loss",
            "official_source_unchanged": True,
            "data": collect_data_hashes(config, project_root),
            "train_seconds": train_seconds,
            "runtime": runtime.to_mapping(),
        }
        write_json(paths.metadata, metadata)
        if tracker is not None:
            tracker.log_metadata(metadata)
        logger.start(
            {
                "pipeline": "official_reference",
                "device": runtime.gpu_name or str(device),
            }
        )
        for item in summary["history"]:
            epoch = int(item["epoch"])
            logger.event("official_epoch", item, epoch=epoch)
            if tracker is not None:
                tracker.log_reference_epoch(item, epoch=epoch)
        logger.epoch(record, evaluation, best=True)
        finalize_outputs(
            paths,
            config,
            record,
            inference,
            audit,
            logger,
            project_root,
            publish_summary=request.publish_summary,
        )
        logger.event("run_completed", {"best_dev_loss": record.dev_loss})
        logger.flush_console()
        rich_log = paths.console_log.read_text(encoding="utf-8")
        paths.console_log.write_text(captured.getvalue() + rich_log, encoding="utf-8")
        entries = finalize_artifact_manifest(paths)
        if tracker is not None:
            tracker.finish(paths, evaluation, audit, entries)
        return TrainResult(paths, record.epoch, record.micro_f1)
    except Exception as error:
        write_json(paths.status, {"status": "failed", "error": str(error)})
        logger.event("run_failed", {"error_type": type(error).__name__, "error": str(error)})
        paths.console_log.write_text(captured.getvalue(), encoding="utf-8")
        if tracker is not None:
            with suppress(Exception):
                tracker.fail(paths, error)
        raise
