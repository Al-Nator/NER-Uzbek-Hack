from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from uzner.experiments.mlflow_transfer import (
    TRANSFER_HOST_TAG,
    TRANSFER_SOURCE_TAG,
    TRANSFER_STATUS_TAG,
    MlflowTransferRequest,
    transfer_mlflow_run,
)


def _tracking_uri(path: Path) -> str:
    """Строит SQLite tracking URI внутри pytest tmp."""
    return f"sqlite:///{path.resolve()}"


@pytest.mark.parametrize("terminal_status", ["FINISHED", "KILLED"])
def test_transfer_copies_full_metric_history_params_tags_and_artifacts(
    tmp_path: Path, terminal_status: str
) -> None:
    """Перенос сохраняет всю лёгкую историю и идемпотентен."""
    source_uri = _tracking_uri(tmp_path / "source.db")
    destination_uri = _tracking_uri(tmp_path / "destination.db")
    source = MlflowClient(tracking_uri=source_uri)
    source_experiment = source.create_experiment(
        "remote",
        artifact_location=(tmp_path / "source-artifacts").resolve().as_uri(),
    )
    source_run = source.create_run(source_experiment, run_name="s22_a100")
    source_run_id = source_run.info.run_id
    source.log_param(source_run_id, "encoder", "xlm-roberta-large")
    source.set_tag(source_run_id, "scheme", "BIOES")
    source.log_metric(source_run_id, "eval/micro/f1", 0.81, timestamp=1000, step=1)
    source.log_metric(source_run_id, "eval/micro/f1", 0.91, timestamp=2000, step=2)
    source.set_terminated(source_run_id, status=terminal_status, end_time=3000)

    run_root = tmp_path / "run"
    (run_root / "metrics").mkdir(parents=True)
    (run_root / "metrics" / "dev.json").write_text('{"micro_f1": 0.91}\n', encoding="utf-8")
    (run_root / "status.json").write_text('{"status": "complete"}\n', encoding="utf-8")
    request = MlflowTransferRequest(
        source_uri=source_uri,
        source_run_id=source_run_id,
        source_host="alnator",
        destination_uri=destination_uri,
        destination_experiment="main",
        destination_artifact_location=(tmp_path / "destination-artifacts").resolve().as_uri(),
        run_root=run_root,
    )

    first = transfer_mlflow_run(request)
    second = transfer_mlflow_run(request)

    assert first.imported is True
    assert first.metric_points == 2
    assert second.imported is False
    assert second.destination_run_id == first.destination_run_id
    destination = MlflowClient(tracking_uri=destination_uri)
    copied = destination.get_run(first.destination_run_id)
    assert copied.info.status == terminal_status
    assert copied.data.params["encoder"] == "xlm-roberta-large"
    assert copied.data.tags["scheme"] == "BIOES"
    assert copied.data.tags[TRANSFER_SOURCE_TAG] == source_run_id
    assert copied.data.tags[TRANSFER_HOST_TAG] == "alnator"
    assert copied.data.tags[TRANSFER_STATUS_TAG] == "complete"
    history = destination.get_metric_history(first.destination_run_id, "eval/micro/f1")
    assert [(point.step, point.value) for point in history] == [(1, 0.81), (2, 0.91)]
    assert destination.list_artifacts(first.destination_run_id, "metrics")[0].path == (
        "metrics/dev.json"
    )


def test_local_publication_preserves_source_link_and_is_idempotent(tmp_path: Path) -> None:
    """Сводка и отдельный local ID корректны даже до загрузки checkpoints."""
    import csv
    import json
    from dataclasses import asdict
    from pathlib import PurePosixPath

    from uzner.config import load_experiment_config
    from uzner.experiments.artifacts import write_resolved_config
    from uzner.experiments.logging import EpochRecord
    from uzner.experiments.remote import RemoteWorkspace
    from uzner.experiments.remote_sync import _publish_local_result

    uri = _tracking_uri(tmp_path / "local.db")
    client = MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment("local", artifact_location=(tmp_path / "art").as_uri())
    run = client.create_run(experiment)
    root = tmp_path / "runs/example"
    (root / "logs").mkdir(parents=True)
    (root / "logs/mlflow_run_id.txt").write_text("remote-id\n")
    (root / ".checkpoint-transfer-pending").touch()
    (root / "status.json").write_text(json.dumps({"status": "complete", "best_epoch": 2}))
    config = load_experiment_config(
        Path("configs/experiments/s22_xlmr_large_bioes_constrained.yaml")
    )
    write_resolved_config(root / "resolved_config.yaml", config)
    values = {key: 0.0 for key in EpochRecord.__dataclass_fields__}
    values.update(epoch=2, micro_f1=0.9)
    record = EpochRecord(**values)
    with (root / "logs/history.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values))
        writer.writeheader()
        writer.writerow(asdict(record))
    workspace = RemoteWorkspace(
        local_root=tmp_path, host="host", remote_root=PurePosixPath("/repo")
    )
    for _repeat in range(2):
        _publish_local_result(workspace, root, run.info.run_id, uri)
    assert (root / "logs/mlflow_run_id.txt").read_text() == "remote-id\n"
    assert (root / "logs/mlflow_local_run_id.txt").read_text().strip() == run.info.run_id
    rows = list(csv.DictReader((tmp_path / "reports/experiments.csv").open()))
    assert len(rows) == 1
    assert rows[0]["micro_f1"] == "0.9"
    assert rows[0]["run_path"] == str(root)
    assert client.get_run(run.info.run_id).data.tags["uzner.transfer.checkpoints"] == "pending"
