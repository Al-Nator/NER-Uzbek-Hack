"""Воспроизводимая CPU-оценка фиксированного ансамбля готовых GPU-предсказаний."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import mlflow
import yaml

from uzner.data.io import load_documents, read_jsonl, sha256_file
from uzner.domain import Prediction
from uzner.evaluation.slices import evaluate_detailed
from uzner.evaluation.span_vote import SpanVoteConfig, majority_vote
from uzner.experiments.artifacts import finalize_root_manifest, write_json, write_jsonl
from uzner.experiments.mlflow_tracking import flatten_numeric
from uzner.experiments.source_snapshot import capture_source


def main() -> None:
    """Записывает exact-метрики и provenance, не создавая новых модельных весов."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    if set(raw) != {"run_id", "sources"}:
        raise ValueError("Неизвестные поля конфига span vote")
    config = SpanVoteConfig(tuple(raw["sources"]))
    if Path(raw["run_id"]).name != raw["run_id"]:
        raise ValueError("run_id должен быть простым именем")
    root = Path("runs") / raw["run_id"]
    if root.exists():
        raise FileExistsError(root)
    files = [Path("runs") / source / "predictions/dev.jsonl" for source in config.sources]
    data = Path("ner_uz_hackathon_participant/data")
    gold = tuple(load_documents((("dev", data / "dev.jsonl"),)))
    train = tuple(load_documents((("train", data / "train.jsonl"),)))
    sources = tuple(tuple(Prediction.from_mapping(r) for r in read_jsonl(p)) for p in files)
    started = time.perf_counter()
    predictions = majority_vote(gold, sources)
    seconds = time.perf_counter() - started
    report = evaluate_detailed(gold, predictions, train, {})
    write_json(root / "resolved_config.json", {**asdict(config), "rule": "exact span 2 of 3"})
    write_json(
        root / "metadata.json",
        {
            "kind": "posthoc_ensemble",
            "fit_on_dev": False,
            "selection_caveat": "source models were previously selected on this dev",
            "input_sha256": {
                str(p): sha256_file(p) for p in [*files, data / "dev.jsonl", data / "train.jsonl"]
            },
            "vote_seconds_without_encoder_inference": seconds,
        },
    )
    write_json(root / "metrics/dev.json", report.overall.to_mapping())
    write_json(root / "metrics/slices.json", report.to_mapping())
    write_jsonl(root / "metrics/errors.jsonl", (e.to_mapping() for e in report.errors.records))
    write_jsonl(root / "predictions/dev.jsonl", (p.to_mapping() for p in predictions))
    capture_source(Path.cwd(), root / "environment")
    mlflow.set_tracking_uri("sqlite:///" + str(Path("mlruns/mlflow.db").resolve()))
    mlflow.set_experiment("uzner-first-series")
    with mlflow.start_run(run_name=root.name, tags={"uzner.kind": "posthoc_ensemble"}) as active:
        write_json(root / "mlflow.json", {"run_id": active.info.run_id})
        mlflow.log_params({"sources": ",".join(config.sources), "votes_required": 2})
        mlflow.log_metrics(flatten_numeric(report.to_mapping(), prefix="dev"))
        mlflow.log_metrics(flatten_numeric(report.to_mapping(), prefix="best/eval"))
        write_json(
            root / "status.json", {"status": "complete", "exact_micro_f1": report.overall.micro.f1}
        )
        finalize_root_manifest(root, root / "artifact_manifest.json")
        mlflow.log_artifacts(str(root))
    print(json.dumps({"run_id": root.name, "f1": report.overall.micro.f1}))


if __name__ == "__main__":
    main()
