"""Финальные метрики, отчёты и сводка эксперимента."""

from __future__ import annotations

from pathlib import Path

from uzner.config import ExperimentConfig
from uzner.data.io import write_predictions
from uzner.evaluation.tokenizer_audit import TokenizerAudit
from uzner.experiments.artifacts import RunPaths, write_json, write_jsonl
from uzner.experiments.logging import EpochRecord, RunLogger
from uzner.experiments.reporting import (
    render_training_svg,
    update_comparison_csv,
    write_run_report,
)
from uzner.training.inference import InferenceResult
from uzner.training.loop import TrainEpochResult


def make_epoch_record(
    epoch: int,
    trained: TrainEpochResult,
    inference: InferenceResult,
    learning_rate: float,
) -> EpochRecord:
    """Собирает плоскую CSV-строку из train/eval итогов."""
    evaluation = inference.evaluation.overall
    return EpochRecord(
        epoch=epoch,
        train_loss=trained.loss,
        dev_loss=inference.loss,
        micro_precision=evaluation.micro.precision,
        micro_recall=evaluation.micro.recall,
        micro_f1=evaluation.micro.f1,
        macro_f1=evaluation.macro_f1,
        org_f1=evaluation.by_label["ORG"].f1,
        name_f1=evaluation.by_label["NAME"].f1,
        geo_f1=evaluation.by_label["GEO"].f1,
        learning_rate=learning_rate,
        mean_gradient_norm=trained.mean_gradient_norm,
        train_seconds=trained.seconds,
        eval_seconds=inference.seconds,
        train_tokens_per_second=trained.tokens_per_second,
        eval_documents_per_second=inference.documents_per_second,
        peak_gpu_memory_gib=trained.peak_gpu_memory_gib,
    )


def write_epoch_metrics(
    paths: RunPaths,
    epoch: int,
    inference: InferenceResult,
) -> None:
    """Сохраняет лёгкую диагностику эпохи без дублирования predictions."""
    write_json(
        paths.metrics.parent / "epochs" / f"epoch_{epoch:02d}.json",
        inference.evaluation.to_mapping(),
    )


def finalize_outputs(
    paths: RunPaths,
    config: ExperimentConfig,
    best_record: EpochRecord,
    inference: InferenceResult,
    audit: TokenizerAudit,
    logger: RunLogger,
    project_root: Path,
    *,
    publish_summary: bool,
) -> None:
    """Пишет predictions, метрики, график, отчёт и status."""
    write_predictions(paths.predictions, inference.predictions)
    write_json(paths.metrics, inference.evaluation.overall.to_mapping())
    write_json(paths.slice_metrics, inference.evaluation.to_mapping())
    write_json(paths.error_summary, inference.evaluation.errors.to_mapping())
    write_jsonl(
        paths.errors,
        (record.to_mapping() for record in inference.evaluation.errors.records),
    )
    write_json(paths.tokenizer_audit, audit.to_mapping())
    render_training_svg(logger.history, paths.training_plot)
    write_run_report(paths.report, config, best_record, inference.evaluation, audit)
    write_json(
        paths.status,
        {
            "status": "complete",
            "best_epoch": best_record.epoch,
            "best_micro_f1": best_record.micro_f1,
        },
    )
    if publish_summary:
        update_comparison_csv(
            project_root / "reports/experiments.csv",
            config,
            best_record,
            paths.root,
        )
