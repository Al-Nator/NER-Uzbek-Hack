"""Финальное обучение на train+dev без validation и выбора по F1."""

from dataclasses import asdict, dataclass, replace
from math import ceil
from pathlib import Path

from transformers import get_linear_schedule_with_warmup

from uzner.config import ExperimentConfig, load_experiment_config
from uzner.domain import Document
from uzner.experiments.artifacts import (
    finalize_artifact_manifest,
    prepare_run_paths,
    write_json,
    write_resolved_config,
)
from uzner.experiments.logging import RunLogger
from uzner.experiments.mlflow_tracking import start_mlflow_run
from uzner.training import checkpoint, runtime
from uzner.training.data_setup import collect_data_hashes, load_split, make_loader
from uzner.training.final_support import final_features, initialize_final, publish_final_alias
from uzner.training.loop import train_epoch
from uzner.training.optimizer import make_optimizer


@dataclass(frozen=True)
class FinalFitRequest:
    """Явный режим финального обучения и изолированного smoke-теста."""

    config_path: Path
    project_root: Path
    smoke_output: Path | None = None
    document_limit: int | None = None
    selected_epoch: int | None = None
    allow_research_initial: bool = False


def final_documents(
    config: ExperimentConfig, root: Path, limit: int | None = None
) -> tuple[Document, ...]:
    """Объединяет train/dev, запрещая повторяющиеся идентификаторы."""
    train = load_split(config, root, "train", limit)
    dev = load_split(config, root, "dev", limit)
    documents = train + dev
    if len({d.hash for d in documents}) != len(documents):
        raise ValueError("В train+dev повторяются hash")
    return documents


def run_final_fit(request: FinalFitRequest) -> Path:
    """Обучает фиксированное число эпох; никогда не считает dev-F1."""
    root = request.project_root.resolve()
    config = load_experiment_config(request.config_path)
    if config.training.early_stopping_patience:
        raise ValueError("Final fit требует early_stopping_patience=0")
    selected_epoch = (
        config.training.epochs if request.selected_epoch is None else request.selected_epoch
    )
    if not 1 <= selected_epoch <= config.training.epochs:
        raise ValueError("Фиксированная эпоха должна входить в бюджет scheduler")
    if request.document_limit is not None and request.smoke_output is None:
        raise ValueError("Ограничение документов разрешено только в smoke")
    if request.smoke_output is not None:
        config = replace(config, output_root=str(request.smoke_output.resolve()))
    elif not config.training.require_gpu:
        raise ValueError("Полный final fit требует GPU")
    documents = final_documents(config, root, request.document_limit)
    paths = prepare_run_paths(config, project_root=root)
    write_resolved_config(paths.resolved_config, config)
    write_json(paths.status, {"status": "running", "kind": "final_fit"})
    logger = RunLogger(config.run_id, paths)
    tracker = None
    try:
        tracker = start_mlflow_run(
            config, paths, root, enabled=request.smoke_output is None, resume=False
        )
        logger.attach_tracker(tracker)
        if tracker:
            tracker.mlflow.set_tags(
                {
                    "uzner.kind": "final_fit",
                    "uzner.selection": "fixed_last_epoch",
                    "uzner.independent_validation": "false",
                }
            )
        device = runtime.resolve_device(config.training.require_gpu)
        runtime.set_reproducible_seed(config.training.seed)
        model, tokenizer = initialize_final(
            config, root, device, allow_research_initial=request.allow_research_initial
        )
        runtime.validate_context_length(model.encoder, config)
        if config.training.gradient_checkpointing:
            model.enable_gradient_checkpointing()
        features = final_features(documents, tokenizer, config)
        loader = make_loader(
            features,
            tokenizer,
            batch_size=config.training.batch_size,
            shuffle=True,
            seed=config.training.seed,
            num_workers=config.training.num_workers,
        )
        optimizer = make_optimizer(model, config.training)
        steps = ceil(len(loader) / config.training.gradient_accumulation_steps)
        total = steps * config.training.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total * config.training.warmup_ratio),
            num_training_steps=total,
        )
        info = runtime.collect_runtime_info(root, device)
        runtime.capture_environment(paths, root, info)
        metadata = {
            "kind": "final_fit",
            "selection": "fixed_last_epoch",
            "selected_epoch": selected_epoch,
            "scheduler_epochs": config.training.epochs,
            "independent_validation": False,
            "allow_research_initial": request.allow_research_initial,
            "data": collect_data_hashes(config, root),
            "data_usage": "train and dev both train",
            "train_documents": len(documents),
            "train_windows": len(features),
            "runtime": info.to_mapping(),
        }
        if config.training.initial_checkpoint:
            from uzner.training.warm_start import initial_checkpoint_metadata

            metadata["initial_checkpoint"] = initial_checkpoint_metadata(
                root / config.training.initial_checkpoint
            )
        write_json(paths.metadata, metadata)
        if tracker:
            tracker.log_metadata(metadata)
        logger.start(
            {
                "kind": "final_fit",
                "documents": len(documents),
                "windows": len(features),
                "epochs": selected_epoch,
                "selection": "last epoch; no validation",
            }
        )
        global_step = 0
        for epoch in range(1, selected_epoch + 1):
            trained = train_epoch(
                model,
                loader,
                optimizer,
                scheduler,
                device,
                config.training,
                logger,
                epoch=epoch,
                global_step=global_step,
            )
            global_step = trained.global_step
            values = asdict(trained)
            write_json(paths.root / "metrics" / f"train_epoch_{epoch:02d}.json", values)
            logger.event("train_epoch_completed", values, epoch=epoch, step=global_step)
            logger.console.print(f"Epoch {epoch}/{config.training.epochs}: {values}")
            if tracker:
                tracker.mlflow.log_metrics(
                    {f"train_epoch/{k}": float(v) for k, v in values.items()}, step=epoch
                )
            state = checkpoint.TrainerState(epoch, global_step, -1.0, epoch, 0)
            checkpoint.save_checkpoint(
                paths.last_checkpoint, model, tokenizer, optimizer, scheduler, state, config
            )
        # best — совместимое имя фиксированного финального checkpoint, не выбор по dev.
        publish_final_alias(paths.last_checkpoint, paths.best_checkpoint)
        write_json(
            paths.status,
            {
                "status": "complete",
                "kind": "final_fit",
                "selected_epoch": state.epoch,
                "selection": "fixed_last_epoch",
                "independent_validation": False,
            },
        )
        logger.flush_console()
        finalize_artifact_manifest(paths)
        if tracker:
            for name in (
                "resolved_config.yaml",
                "metadata.json",
                "status.json",
                "artifact_manifest.json",
            ):
                tracker.mlflow.log_artifact(str(paths.root / name), artifact_path="run")
            for name in ("logs", "metrics", "environment"):
                tracker.mlflow.log_artifacts(str(paths.root / name), artifact_path=name)
            tracker.mlflow.end_run(status="FINISHED")
        return paths.root
    except Exception as error:
        write_json(paths.status, {"status": "failed", "kind": "final_fit", "error": str(error)})
        logger.flush_console()
        if tracker:
            tracker.fail(paths, error)
        raise
