"""Публикация полных serving-измерений в MLflow без весов, engine и smoke."""

import argparse
import json
import os
from pathlib import Path

import mlflow

from uzner.data.io import sha256_file
from uzner.experiments.artifacts import write_json
from uzner.experiments.mlflow_tracking import flatten_numeric


def publish(path: Path, bundle: Path) -> str | None:
    """Сохраняет полный отчёт однократно, проверяя его неизменность при повторе."""
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text("utf-8"))
    if summary.get("pilot"):
        return None
    digest = sha256_file(summary_path)
    link = path / "mlflow.json"
    if link.exists():
        record = json.loads(link.read_text("utf-8"))
        if record["summary_sha256"] != digest:
            raise ValueError(f"Изменён уже опубликованный benchmark: {path}")
        return record["run_id"]
    with mlflow.start_run(
        run_name=path.name,
        tags={"uzner.kind": "serving_benchmark", "uzner.measurement": summary["kind"]},
    ) as active:
        parameters = {
            "variant": path.name,
            "input_sha256": summary["input_sha256"],
            "summary_sha256": digest,
            "bundle_sha256": sha256_file(bundle / "manifest.json"),
            "gpu_driver": summary["hardware"]["gpu"],
            "runtime_config": json.dumps(summary["config"]),
            "response_cache": False,
        }
        mlflow.log_params(parameters)
        mlflow.log_metrics(flatten_numeric(summary, prefix="serving"))
        for artifact in sorted(path.iterdir()):
            if artifact.suffix in {".json", ".jsonl"} and artifact.name != "mlflow.json":
                mlflow.log_artifact(str(artifact))
        mlflow.log_artifact(str(bundle / "manifest.json"), "bundle")
        run_id = active.info.run_id
    write_json(link, {"run_id": run_id, "summary_sha256": digest})
    return run_id


def main() -> None:
    """Импортирует результаты A100 в локальный backend, не дублируя GPU-работу."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/serving/benchmarks"))
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/serving/s62-c02-v1"))
    parser.add_argument("--experiment", default="uzner-serving-a100")
    args = parser.parse_args()
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "sqlite:///" + str(Path("mlruns/mlflow.db").resolve()))
    )
    mlflow.set_experiment(args.experiment)
    for summary in sorted(args.root.glob("*/summary.json")):
        run_id = publish(summary.parent, args.bundle)
        if run_id:
            print(f"{summary.parent.name}: {run_id}")


if __name__ == "__main__":
    main()
