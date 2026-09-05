"""Оценивает фиксированный checkpoint на dev при заранее заданных порогах."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import yaml

from uzner.config import ExperimentConfig
from uzner.data.io import sha256_file
from uzner.data.spans import with_span_targets
from uzner.data.windows import build_window_features
from uzner.experiments.artifacts import (
    finalize_artifact_manifest,
    prepare_run_paths,
    write_json,
    write_jsonl,
    write_resolved_config,
)
from uzner.experiments.mlflow_tracking import flatten_numeric, start_mlflow_run
from uzner.training.checkpoint import load_model_checkpoint
from uzner.training.data_setup import collect_data_hashes, load_split, make_loader
from uzner.training.inference import run_inference
from uzner.training.runtime import (
    capture_environment,
    collect_runtime_info,
    resolve_device,
    set_reproducible_seed,
)


def main() -> None:
    """Повторяет полный inference, не фильтруя уже обрезанные predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.35, 0.4, 0.45, 0.5, 0.55])
    args = parser.parse_args()
    root = Path.cwd()
    source = args.source_run.resolve()
    config = ExperimentConfig.from_mapping(
        yaml.safe_load((source / "resolved_config.yaml").read_text())
    )
    if config.model.head != "global_pointer":
        raise ValueError("Sweep предназначен для GlobalPointer")
    if len(set(args.thresholds)) != len(args.thresholds):
        raise ValueError("Пороги должны быть уникальны")
    for threshold in args.thresholds:
        replace(config.model, span_threshold=threshold)
    device = resolve_device(True)
    set_reproducible_seed(config.training.seed)
    loaded = load_model_checkpoint(source / "checkpoints/best", config, device)
    train = load_split(config, root, "train", None)
    dev = load_split(config, root, "dev", None)
    features = with_span_targets(
        build_window_features(
            dev,
            loaded.tokenizer,
            config.tokenization,
            config.model.tag_scheme,
            with_labels=False,
        ),
        dev,
    )
    loader = make_loader(
        features,
        loaded.tokenizer,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        seed=config.training.seed,
        num_workers=4,
    )
    fingerprint = {
        str(path.relative_to(source)): sha256_file(path)
        for path in (source / "checkpoints/best").rglob("*.safetensors")
    }
    for threshold in args.thresholds:
        trial = replace(
            config,
            run_id=f"s32_threshold_{threshold:.2f}_v1",
            model=replace(config.model, span_threshold=threshold),
        )
        paths = prepare_run_paths(trial, project_root=root)
        write_resolved_config(paths.resolved_config, trial)
        write_json(paths.status, {"status": "running", "kind": "dev_threshold_selection"})
        tracker = start_mlflow_run(trial, paths, root, enabled=True, resume=False)
        try:
            tracker.mlflow.set_tags(
                {"uzner.kind": "dev_threshold_selection", "uzner.parent_run": source.name}
            )
            metadata = {
                "source_run": source.name,
                "checkpoint_epoch": loaded.state.epoch,
                "checkpoint_sha256": fingerprint,
                "data": collect_data_hashes(config, root),
                "selection_split": "dev",
                "training_performed": False,
            }
            write_json(paths.metadata, metadata)
            capture_environment(paths, root, collect_runtime_info(root, device))
            loaded.model.model_config = trial.model
            result = run_inference(
                loaded.model, loader, features, dev, train, device, bf16=config.training.bf16
            )
            write_json(paths.metrics, result.evaluation.overall.to_mapping())
            write_json(paths.slice_metrics, result.evaluation.to_mapping())
            write_jsonl(paths.predictions, (item.to_mapping() for item in result.predictions))
            tracker.mlflow.log_metrics(
                flatten_numeric(result.evaluation.to_mapping(), prefix="eval")
            )
            write_json(
                paths.status,
                {
                    "status": "complete",
                    "kind": "dev_threshold_selection",
                    "micro_f1": result.evaluation.overall.micro.f1,
                },
            )
            finalize_artifact_manifest(paths)
            for name in (
                "resolved_config.yaml",
                "metadata.json",
                "status.json",
                "artifact_manifest.json",
            ):
                tracker.mlflow.log_artifact(str(paths.root / name), "run")
            for name in ("metrics", "predictions", "environment"):
                tracker.mlflow.log_artifacts(str(paths.root / name), name)
            tracker.mlflow.end_run()
            print(
                json.dumps(
                    {"threshold": threshold, **result.evaluation.overall.micro.to_mapping()}
                ),
                flush=True,
            )
        except Exception:
            write_json(paths.status, {"status": "failed"})
            tracker.mlflow.end_run(status="FAILED")
            raise


if __name__ == "__main__":
    main()
