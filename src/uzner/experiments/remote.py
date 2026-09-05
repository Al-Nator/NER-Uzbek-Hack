"""Запуск экспериментов на SSH-GPU и атомарный возврат артефактов."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from uzner.config import load_experiment_config, with_run_suffix
from uzner.data.io import sha256_file
from uzner.experiments.series import load_series

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_DATA_FILES = (
    "ner_uz_hackathon_participant/data/train.jsonl",
    "ner_uz_hackathon_participant/data/dev.jsonl",
)


@dataclass(frozen=True, slots=True)
class RemoteWorkspace:
    """Хранит явные адреса удалённого GPU-окружения."""

    local_root: Path
    host: str = "alnator"
    remote_root: PurePosixPath = PurePosixPath("/home/danya/NER-Uzbek-Hack")
    remote_mlflow_port: int = 5000
    local_mlflow_port: int = 5001
    remote_experiment: str = "uzner-second-series-a100"
    destination_experiment: str = "uzner-first-series"


@dataclass(frozen=True, slots=True)
class RemoteCheck:
    """Описывает сверенные commit, данные и GPU."""

    commit: str
    gpu: str
    data_hashes: tuple[str, ...]


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Выполняет одну команду без неявного shell."""
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def ssh_command(workspace: RemoteWorkspace, remote: list[str]) -> list[str]:
    """Строит SSH-команду с безопасным shell quoting."""
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        workspace.host,
        shlex.join(remote),
    ]


def _ssh(workspace: RemoteWorkspace, remote: list[str], *, check: bool = True) -> str:
    """Выполняет SSH-команду и возвращает её вывод."""
    return _run(ssh_command(workspace, remote), check=check).stdout.strip()


def selected_run_ids(
    workspace: RemoteWorkspace,
    series_path: Path,
    stage: str,
    suffix: str,
) -> tuple[str, ...]:
    """Разрешает имена run-ов точно как основной series runner."""
    series = load_series((workspace.local_root / series_path).resolve())
    result = []
    for config_name in series.select(stage):
        config = load_experiment_config((workspace.local_root / config_name).resolve())
        result.append(with_run_suffix(config, suffix).run_id)
    return tuple(result)


def prepare_remote(workspace: RemoteWorkspace, *, install: bool = True) -> RemoteCheck:
    """Сверяет commit/data/GPU и синхронизирует uv-среду."""
    local_commit = _run(["git", "rev-parse", "HEAD"], cwd=workspace.local_root).stdout.strip()
    remote_commit = _ssh(
        workspace,
        ["git", "-C", str(workspace.remote_root), "rev-parse", "HEAD"],
    )
    if local_commit != remote_commit:
        raise RuntimeError(f"Commit-ы различаются: local={local_commit}, remote={remote_commit}")
    hashes = []
    for relative in _DATA_FILES:
        local_hash = sha256_file(workspace.local_root / relative)
        output = _ssh(workspace, ["sha256sum", str(workspace.remote_root / relative)])
        remote_hash = output.split(maxsplit=1)[0]
        if local_hash != remote_hash:
            raise RuntimeError(f"Хэш данных не совпал: {relative}")
        hashes.append(local_hash)
    if install:
        _ssh(
            workspace,
            [
                "uv",
                "--directory",
                str(workspace.remote_root),
                "sync",
                "--extra",
                "train",
                "--group",
                "dev",
                "--frozen",
            ],
        )
    gpu = _ssh(
        workspace,
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader",
        ],
    )
    return RemoteCheck(local_commit, gpu, tuple(hashes))


def preflight_remote(
    workspace: RemoteWorkspace,
    *,
    series_path: Path,
    stage: str,
) -> str:
    """Загружает encoder и проверяет BF16 forward на удалённой GPU."""
    return _ssh(
        workspace,
        [
            "uv",
            "--directory",
            str(workspace.remote_root),
            "run",
            "python",
            "scripts/preflight_models.py",
            "--series",
            str(series_path),
            "--stage",
            stage,
            "--project-root",
            str(workspace.remote_root),
        ],
    )


def _session_names(workspace: RemoteWorkspace) -> set[str]:
    """Читает имена активных zellij-сессий."""
    output = _ssh(workspace, ["zellij", "list-sessions", "--short"], check=False)
    return {line.strip() for line in output.splitlines() if line.strip()}


def start_remote_mlflow(workspace: RemoteWorkspace, *, session: str = "uzner-mlflow") -> bool:
    """Поднимает независимый MLflow в фоновой zellij-сессии."""
    health = f"http://127.0.0.1:{workspace.remote_mlflow_port}/health"
    if _ssh(workspace, ["curl", "--fail", "--silent", health], check=False):
        return False
    if session not in _session_names(workspace):
        _ssh(workspace, ["zellij", "attach", "--create-background", session])
    _ssh(
        workspace,
        [
            "zellij",
            "--session",
            session,
            "run",
            "--cwd",
            str(workspace.remote_root),
            "--name",
            "mlflow",
            "--",
            "uv",
            "run",
            "--extra",
            "train",
            "mlflow",
            "server",
            "--backend-store-uri",
            "sqlite:///mlruns/mlflow.db",
            "--default-artifact-root",
            "./mlruns/artifacts",
            "--host",
            "127.0.0.1",
            "--port",
            str(workspace.remote_mlflow_port),
        ],
    )
    for _attempt in range(30):
        time.sleep(1)
        if _ssh(workspace, ["curl", "--fail", "--silent", health], check=False):
            return True
    raise RuntimeError("Удалённый MLflow не прошёл health-check за 30 секунд")


def start_remote_training(
    workspace: RemoteWorkspace,
    *,
    series_path: Path,
    stage: str,
    suffix: str,
    session: str,
) -> tuple[str, ...]:
    """Запускает выбранный этап в отдельной zellij-сессии."""
    if not _RUN_ID.fullmatch(session):
        raise ValueError("Небезопасное имя zellij-сессии")
    if session in _session_names(workspace):
        raise FileExistsError(f"Zellij-сессия уже существует: {session}")
    run_ids = selected_run_ids(workspace, series_path, stage, suffix)
    for run_id in run_ids:
        remote_run = workspace.remote_root / "runs" / run_id
        exists = _run(ssh_command(workspace, ["test", "-e", str(remote_run)]), check=False)
        if exists.returncode == 0:
            raise FileExistsError(f"Удалённый run уже существует: {run_id}")
    _ssh(workspace, ["zellij", "attach", "--create-background", session])
    _ssh(
        workspace,
        [
            "zellij",
            "--session",
            session,
            "run",
            "--cwd",
            str(workspace.remote_root),
            "--name",
            "training",
            "--",
            "env",
            f"UZNER_MLFLOW_EXPERIMENT={workspace.remote_experiment}",
            "uv",
            "run",
            "python",
            "scripts/run_series.py",
            "--series",
            str(series_path),
            "--stage",
            stage,
            "--run-suffix",
            suffix,
        ],
    )
    return run_ids


def remote_status(workspace: RemoteWorkspace, session: str) -> str:
    """Возвращает zellij panes, GPU и статусы удалённых run-ов."""
    panes = _ssh(
        workspace,
        [
            "zellij",
            "--session",
            session,
            "action",
            "list-panes",
            "--json",
            "--all",
            "--state",
            "--command",
        ],
        check=False,
    )
    gpu = _ssh(
        workspace,
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ],
    )
    if panes.startswith("["):
        values = [item for item in json.loads(panes) if not item["is_plugin"]]
        panes = "; ".join(
            f"{item['title']} ({'exited' if item['exited'] else 'running'}): "
            f"{item.get('terminal_command') or item.get('pane_command')}"
            for item in values
        )
    return f"GPU: {gpu}\nZellij: {panes or 'сессия не найдена'}"
