"""Обучение предлагающих моделей только на внутреннем train-разбиении."""

import argparse
import json
import os
from pathlib import Path

from mlflow.tracking import MlflowClient

from uzner.data.io import sha256_file
from uzner.experiments.artifacts import finalize_root_manifest, write_json
from uzner.training.engine import TrainRequest, train_experiment


def main() -> None:
    """Запускает три независимых pretrained-модели и помечает внутреннюю оценку."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = train_experiment(TrainRequest(args.config, Path.cwd(), publish_summary=False))
    root = result.paths.root
    status = json.loads((root / "status.json").read_text())
    status.update(kind="reranker_proposer", evaluation_split="proposer_valid_not_official_dev")
    write_json(root / "status.json", status)
    split = Path("artifacts/reranker/s7_holdout_v1/manifest.json")
    write_json(
        root / "holdout_protocol.json",
        {
            "manifest_sha256": sha256_file(split),
            "source": str(split),
            "initialization": "pretrained; no full-train NER checkpoint",
            "validation": "proposer_valid only; meta and official dev excluded",
        },
    )
    finalize_root_manifest(root, root / "artifact_manifest.json")
    client = MlflowClient(
        tracking_uri=os.getenv(
            "MLFLOW_TRACKING_URI", "sqlite:///" + str(Path("mlruns/mlflow.db").resolve())
        )
    )
    run_id = (root / "logs/mlflow_run_id.txt").read_text().strip()
    client.set_tag(run_id, "uzner.kind", "reranker_proposer")
    client.set_tag(run_id, "uzner.evaluation_split", "proposer_valid_not_official_dev")
    for name in ("status.json", "holdout_protocol.json", "artifact_manifest.json"):
        client.log_artifact(run_id, str(root / name), "run")


if __name__ == "__main__":
    main()
