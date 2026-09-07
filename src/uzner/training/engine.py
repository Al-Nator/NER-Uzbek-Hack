"""Воспроизводимый train → exact-eval → checkpoint → report контур."""

from __future__ import annotations

import gc
import math
from contextlib import suppress

import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from uzner.data.spans import span_coverage, with_span_targets
from uzner.data.windows import build_window_features
from uzner.evaluation.tokenizer_audit import audit_tokenizer
from uzner.experiments.artifacts import (
    finalize_artifact_manifest,
    prepare_run_paths,
    write_json,
    write_resolved_config,
)
from uzner.experiments.logging import EpochRecord, RunLogger
from uzner.experiments.mlflow_tracking import MlflowTracker, start_mlflow_run
from uzner.models.factory import model_class
from uzner.models.pretrained import resolve_pretrained_snapshot
from uzner.training import checkpoint, finalization, runtime
from uzner.training.data_setup import (
    collect_data_hashes,
    load_split,
    make_loader,
    validate_split_separation,
)
from uzner.training.inference import InferenceResult, run_inference
from uzner.training.loop import train_epoch
from uzner.training.optimizer import make_optimizer
from uzner.training.request_config import resolve_request_config
from uzner.training.requests import TrainRequest, TrainResult
from uzner.training.warm_start import initial_checkpoint_metadata, load_initial_checkpoint


def train_experiment(request: TrainRequest) -> TrainResult:
    """Выполняет один полный конфигурируемый эксперимент."""
    project_root = request.project_root.resolve()
    config = resolve_request_config(request)
    if config.pipeline != "uzner":
        raise ValueError("Official reference запускается отдельным wrapper-ом")
    train_documents = load_split(config, project_root, "train", request.max_train_documents)
    dev_documents = load_split(config, project_root, "dev", request.max_dev_documents)
    validate_split_separation(train_documents, dev_documents)
    paths = prepare_run_paths(config, project_root=project_root, resume=request.resume)
    logger = RunLogger(config.run_id, paths)
    tracker: MlflowTracker | None = None
    write_resolved_config(paths.resolved_config, config)
    write_json(paths.status, {"status": "running"})
    try:
        tracker = start_mlflow_run(
            config,
            paths,
            project_root,
            enabled=request.output_root_override is None,
            resume=request.resume,
        )
        logger.attach_tracker(tracker)
        device = runtime.resolve_device(config.training.require_gpu)
        runtime.set_reproducible_seed(config.training.seed)

        if request.resume:
            loaded = checkpoint.load_model_checkpoint(paths.last_checkpoint, config, device)
            model, tokenizer, state = loaded.model, loaded.tokenizer, loaded.state
            del loaded
        elif config.training.initial_checkpoint:
            loaded = load_initial_checkpoint(
                project_root / config.training.initial_checkpoint, config, device
            )
            model, tokenizer, state = loaded.model, loaded.tokenizer, loaded.state
            del loaded
        else:
            snapshot = resolve_pretrained_snapshot(config.encoder)
            tokenizer = AutoTokenizer.from_pretrained(
                snapshot.path,
                trust_remote_code=config.encoder.trust_remote_code,
                use_fast=True,
                local_files_only=True,
                fix_mistral_regex=False,
            )
            if not tokenizer.is_fast:
                raise ValueError("Exact offsets требуют fast tokenizer")
            model = (
                model_class(config.model)
                .from_pretrained(
                    config.encoder,
                    config.model,
                    source=snapshot.path,
                )
                .to(device)
            )
            state = checkpoint.TrainerState(0, 0, -1.0, 0, 0)
        runtime.validate_context_length(model.encoder, config)
        if config.training.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        train_features = build_window_features(
            train_documents,
            tokenizer,
            config.tokenization,
            config.model.tag_scheme,
            with_labels=config.model.architecture == "token_tagging",
        )
        dev_features = build_window_features(
            dev_documents,
            tokenizer,
            config.tokenization,
            config.model.tag_scheme,
            with_labels=config.model.architecture == "token_tagging",
        )
        if config.model.architecture == "span":
            train_features = with_span_targets(train_features, train_documents)
            dev_features = with_span_targets(dev_features, dev_documents)
            write_json(
                paths.metrics.parent / "span_coverage.json",
                {
                    "train": span_coverage(train_features, train_documents),
                    "dev": span_coverage(dev_features, dev_documents),
                },
            )
        audit = audit_tokenizer(dev_documents, tokenizer, config.encoder, config.tokenization)
        train_loader = make_loader(
            train_features,
            tokenizer,
            batch_size=config.training.batch_size,
            shuffle=True,
            seed=config.training.seed,
            num_workers=config.training.num_workers,
        )
        dev_loader = make_loader(
            dev_features,
            tokenizer,
            batch_size=config.training.eval_batch_size,
            shuffle=False,
            seed=config.training.seed,
            num_workers=config.training.num_workers,
        )
        optimizer = make_optimizer(model, config.training)
        updates_per_epoch = math.ceil(
            len(train_loader) / config.training.gradient_accumulation_steps
        )
        total_updates = updates_per_epoch * config.training.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_updates * config.training.warmup_ratio),
            num_training_steps=total_updates,
        )
        if request.resume:
            checkpoint.restore_training_state(paths.last_checkpoint, optimizer, scheduler)

        runtime_info = runtime.collect_runtime_info(project_root, device)
        runtime.capture_environment(paths, project_root, runtime_info)
        metadata = {
            "schema_version": 1,
            "run_id": config.run_id,
            "data": collect_data_hashes(config, project_root),
            "train_documents": len(train_documents),
            "dev_documents": len(dev_documents),
            "train_windows": len(train_features),
            "dev_windows": len(dev_features),
            "model_parameters_total": sum(parameter.numel() for parameter in model.parameters()),
            "model_parameters_trainable": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "runtime": runtime_info.to_mapping(),
        }
        write_json(paths.metadata, metadata)
        if config.training.initial_checkpoint:
            metadata["initial_checkpoint"] = initial_checkpoint_metadata(
                project_root / config.training.initial_checkpoint
            )
            write_json(paths.metadata, metadata)
        if tracker is not None:
            tracker.log_metadata(metadata)
        logger.start(
            {
                "device": runtime_info.gpu_name or runtime_info.device,
                "encoder": config.encoder.name,
                "tags/head/decoder": (
                    f"{config.model.tag_scheme}/{config.model.head}/{config.model.decoder}"
                ),
                "context/stride": (
                    f"{config.tokenization.max_length}/{config.tokenization.stride}"
                ),
                "documents train/dev": f"{len(train_documents)}/{len(dev_documents)}",
                "windows train/dev": f"{len(train_features)}/{len(dev_features)}",
            }
        )

        best_inference: InferenceResult | None = None
        best_record: EpochRecord | None = None
        for epoch in range(state.epoch + 1, config.training.epochs + 1):
            trained = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                config.training,
                logger,
                epoch=epoch,
                global_step=state.global_step,
            )
            inference = run_inference(
                model,
                dev_loader,
                dev_features,
                dev_documents,
                train_documents,
                device,
                bf16=config.training.bf16,
            )
            record = finalization.make_epoch_record(
                epoch,
                trained,
                inference,
                float(scheduler.get_last_lr()[0]),
            )
            improved = record.micro_f1 > state.best_micro_f1
            state = checkpoint.TrainerState(
                epoch=epoch,
                global_step=trained.global_step,
                best_micro_f1=record.micro_f1 if improved else state.best_micro_f1,
                best_epoch=epoch if improved else state.best_epoch,
                epochs_without_improvement=(
                    0 if improved else state.epochs_without_improvement + 1
                ),
            )
            finalization.write_epoch_metrics(paths, epoch, inference)
            checkpoint.save_checkpoint(
                paths.last_checkpoint,
                model,
                tokenizer,
                optimizer,
                scheduler,
                state,
                config,
            )
            if improved:
                checkpoint.save_checkpoint(
                    paths.best_checkpoint,
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    state,
                    config,
                )
                best_inference, best_record = inference, record
            logger.epoch(record, inference.evaluation, best=improved)
            patience = config.training.early_stopping_patience
            if patience and state.epochs_without_improvement >= patience:
                logger.event("early_stopping", {"patience": patience}, epoch=epoch)
                break

        if best_inference is None or best_record is None:
            # Освобождаем last и optimizer до загрузки best на ту же GPU.
            del model, optimizer, scheduler
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            loaded_best = checkpoint.load_model_checkpoint(paths.best_checkpoint, config, device)
            best_inference = run_inference(
                loaded_best.model,
                dev_loader,
                dev_features,
                dev_documents,
                train_documents,
                device,
                bf16=config.training.bf16,
            )
            matched = [item for item in logger.history if item.epoch == state.best_epoch]
            if not matched:
                raise RuntimeError("В history не найдена best epoch после resume")
            best_record = matched[0]
        finalization.finalize_outputs(
            paths,
            config,
            best_record,
            best_inference,
            audit,
            logger,
            project_root,
            publish_summary=request.publish_summary,
        )
        logger.event("run_completed", {"best_micro_f1": best_record.micro_f1})
        logger.flush_console()
        entries = finalize_artifact_manifest(paths)
        if tracker is not None:
            tracker.finish(paths, best_inference.evaluation, audit, entries)
        return TrainResult(paths, best_record.epoch, best_record.micro_f1)
    except Exception as error:
        write_json(paths.status, {"status": "failed", "error": str(error)})
        logger.event("run_failed", {"error_type": type(error).__name__, "error": str(error)})
        logger.flush_console()
        if tracker is not None:
            with suppress(Exception):
                tracker.fail(paths, error)
        raise
