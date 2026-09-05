"""Возврат артефактов и MLflow-истории с SSH-GPU."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import yaml

from uzner.config import ExperimentConfig
from uzner.data.io import sha256_file
from uzner.experiments.logging import EpochRecord
from uzner.experiments.mlflow_tracking import MlflowSettings
from uzner.experiments.mlflow_transfer import (
    MlflowTransferRequest,
    MlflowTransferResult,
    transfer_mlflow_run,
)
from uzner.experiments.remote import RemoteWorkspace, _run, ssh_command
from uzner.experiments.reporting import update_comparison_csv

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class RemoteSync:
    """Описывает результат возврата одного run."""

    run_root: Path
    copied: bool
    mlflow: MlflowTransferResult


def verify_run_manifest(run_root: Path, *, include_checkpoints: bool = True) -> None:
    """Проверяет размеры и SHA-256 всех файлов из run-манифеста."""
    manifest_path = run_root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["artifacts"]:
        if not include_checkpoints and Path(entry["path"]).parts[0] == "checkpoints":
            continue
        path = run_root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Не найден артефакт: {entry['path']}")
        valid_size = path.stat().st_size == entry["bytes"]
        if not valid_size or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Не совпал артефакт: {entry['path']}")


def remote_run_status(workspace: RemoteWorkspace, run_id: str) -> str | None:
    """Читает статус run-а; до создания каталога возвращает None."""
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("Небезопасный run_id")
    status_path = workspace.remote_root / "runs" / run_id / "status.json"
    result = _run(ssh_command(workspace, ["cat", str(status_path)]), check=False)
    if result.returncode != 0:
        return None
    status = str(json.loads(result.stdout).get("status"))
    if status == "complete":
        manifest = status_path.parent / "artifact_manifest.json"
        exists = _run(ssh_command(workspace, ["test", "-f", str(manifest)]), check=False)
        return "complete" if exists.returncode == 0 else "finalizing"
    return status


def copy_remote_run(
    workspace: RemoteWorkspace, run_id: str, *, include_checkpoints: bool = True
) -> tuple[Path, bool]:
    """Атомарно копирует завершённый run и проверяет манифест."""
    remote_run = workspace.remote_root / "runs" / run_id
    if remote_run_status(workspace, run_id) != "complete":
        raise RuntimeError(f"Run ещё не завершён: {run_id}")
    target = workspace.local_root / "runs" / run_id
    pending = target / ".checkpoint-transfer-pending"
    if target.exists() and (not include_checkpoints or not pending.exists()):
        verify_run_manifest(target, include_checkpoints=include_checkpoints)
        return target, False
    incoming = workspace.local_root / "runs" / ".remote-incoming" / run_id
    incoming.mkdir(parents=True, exist_ok=True)
    # При догрузке используем partial-файлы первого rsync и затем публикуем checkpoints.
    excludes = [] if include_checkpoints else ["--exclude=/checkpoints/"]
    _run(
        [
            "rsync",
            "--archive",
            "--partial",
            "--info=progress2",
            *excludes,
            f"{workspace.host}:{remote_run}/",
            f"{incoming}/",
        ]
    )
    verify_run_manifest(incoming, include_checkpoints=include_checkpoints)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not include_checkpoints:
        # Частичные веса остаются в incoming; пользователю доступны лёгкие артефакты.
        target.mkdir(exist_ok=True)
        _run(["rsync", "--archive", "--exclude=/checkpoints/", f"{incoming}/", f"{target}/"])
        pending.touch()
    elif target.exists():
        (incoming / "checkpoints").replace(target / "checkpoints")
        verify_run_manifest(target)
        pending.unlink(missing_ok=True)
    else:
        incoming.replace(target)
    return target, True


def _url_is_healthy(url: str) -> bool:
    """Проверяет локальный HTTP health endpoint без долгого ожидания."""
    try:
        with urlopen(url, timeout=1) as response:  # noqa: S310
            return response.status == 200
    except OSError:
        return False


@contextmanager
def temporary_mlflow_tunnel(workspace: RemoteWorkspace) -> Iterator[str]:
    """Открывает временный SSH tunnel или переиспользует уже работающий."""
    uri = f"http://127.0.0.1:{workspace.local_mlflow_port}"
    health = f"{uri}/health"
    if _url_is_healthy(health):
        yield uri
        return
    process = subprocess.Popen(_tunnel_command(workspace, batch_mode=True))
    try:
        for _attempt in range(50):
            if process.poll() is not None:
                raise RuntimeError("SSH tunnel завершился до health-check")
            if _url_is_healthy(health):
                yield uri
                return
            time.sleep(0.2)
        raise TimeoutError("MLflow tunnel не открылся за 10 секунд")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def sync_remote_run(
    workspace: RemoteWorkspace, run_id: str, *, include_checkpoints: bool = True
) -> RemoteSync:
    """Копирует run и импортирует его в основной MLflow."""
    run_root, copied = copy_remote_run(workspace, run_id, include_checkpoints=include_checkpoints)
    link = run_root / "logs" / "mlflow_run_id.txt"
    source_run_id = link.read_text(encoding="utf-8").strip()
    settings = MlflowSettings.from_environment(workspace.local_root)
    with temporary_mlflow_tunnel(workspace) as source_uri:
        transferred = transfer_mlflow_run(
            MlflowTransferRequest(
                source_uri=source_uri,
                source_run_id=source_run_id,
                source_host=workspace.host,
                destination_uri=settings.tracking_uri,
                destination_experiment=workspace.destination_experiment,
                destination_artifact_location=(workspace.local_root / "mlruns/artifacts").as_uri(),
                run_root=run_root,
            )
        )
    _publish_local_result(
        workspace, run_root, transferred.destination_run_id, settings.tracking_uri
    )
    return RemoteSync(run_root, copied, transferred)


def _publish_local_result(
    workspace: RemoteWorkspace, run_root: Path, local_run_id: str, tracking_uri: str
) -> None:
    """Связывает локальный MLflow с артефактами и добавляет проверенный результат в CSV."""
    from mlflow.tracking import MlflowClient

    status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
    with (run_root / "logs/history.csv").open(encoding="utf-8") as stream:
        row = next(
            item for item in csv.DictReader(stream) if int(item["epoch"]) == status["best_epoch"]
        )
    record = EpochRecord(
        **{key: int(value) if key == "epoch" else float(value) for key, value in row.items()}
    )
    config = ExperimentConfig.from_mapping(
        yaml.safe_load((run_root / "resolved_config.yaml").read_text(encoding="utf-8"))
    )
    update_comparison_csv(
        workspace.local_root / "reports/experiments.csv", config, record, run_root
    )
    (run_root / "logs/mlflow_local_run_id.txt").write_text(local_run_id + "\n", encoding="utf-8")
    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_tag(local_run_id, "uzner.transfer.local_run_path", str(run_root))
    pending = (run_root / ".checkpoint-transfer-pending").exists()
    client.set_tag(local_run_id, "uzner.transfer.checkpoints", "pending" if pending else "verified")


def _tunnel_command(workspace: RemoteWorkspace, *, batch_mode: bool) -> list[str]:
    """Строит SSH port-forward с явными адресами."""
    forward = f"127.0.0.1:{workspace.local_mlflow_port}:127.0.0.1:{workspace.remote_mlflow_port}"
    command = ["ssh", "-o", "ExitOnForwardFailure=yes"]
    if batch_mode:
        command.extend(["-o", "BatchMode=yes"])
    return [*command, "-N", "-L", forward, workspace.host]


def tunnel_command(workspace: RemoteWorkspace) -> list[str]:
    """Строит команду для долгоживущего MLflow tunnel."""
    return _tunnel_command(workspace, batch_mode=False)
