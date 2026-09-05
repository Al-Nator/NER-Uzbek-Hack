"""CLI для запуска и возврата experiment run-ов с SSH-GPU."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path, PurePosixPath

from uzner.experiments.remote import (
    RemoteWorkspace,
    preflight_remote,
    remote_status,
    selected_run_ids,
    start_remote_mlflow,
    start_remote_training,
)
from uzner.experiments.remote import prepare_remote as prepare_remote_workspace
from uzner.experiments.remote_sync import (
    remote_run_status,
    sync_remote_run,
    tunnel_command,
)

DEFAULT_SERIES = Path("configs/series/second_a100.yaml")
DEFAULT_STAGE = "xlmr-large"
DEFAULT_SUFFIX = "s2-sequence-a100-v2"
DEFAULT_SESSION = "uzner-s22-s23"


def parse_args() -> argparse.Namespace:
    """Разбирает команду удалённого workflow."""
    parser = argparse.ArgumentParser(description="Управлять experiment run-ами на A100")
    parser.add_argument("--host", default="alnator")
    parser.add_argument("--remote-root", default="/home/danya/NER-Uzbek-Hack")
    parser.add_argument("--local-port", type=int, default=5001)
    parser.add_argument("--remote-port", type=int, default=5000)
    parser.add_argument("--remote-experiment", default="uzner-second-series-a100")
    parser.add_argument("--destination-experiment", default="uzner-first-series")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="сверить код/данны и GPU")
    prepare.add_argument("--no-install", action="store_true")
    prepare.add_argument("--preflight", action="store_true")
    _add_selection(prepare)

    start = subparsers.add_parser("start", help="запустить MLflow и обучение в zellij")
    _add_selection(start)
    start.add_argument("--session", default=DEFAULT_SESSION)

    status = subparsers.add_parser("status", help="показать GPU и zellij panes")
    status.add_argument("--session", default=DEFAULT_SESSION)

    attach = subparsers.add_parser("attach", help="подключиться к training zellij")
    attach.add_argument("--session", default=DEFAULT_SESSION)

    subparsers.add_parser("tunnel", help="открыть remote MLflow на localhost")

    sync = subparsers.add_parser("sync", help="вернуть готовые run-ы и MLflow")
    _add_selection(sync)
    sync.add_argument("--run-id", action="append", default=[])
    sync.add_argument(
        "--metrics-only",
        action="store_true",
        help="сначала вернуть лёгкие артефакты и MLflow; веса догрузить обычным sync",
    )

    watch = subparsers.add_parser("watch", help="ждать и забирать run-ы по готовности")
    _add_selection(watch)
    watch.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def _add_selection(parser: argparse.ArgumentParser) -> None:
    """Добавляет общий выбор серии, этапа и suffix."""
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--run-suffix", default=DEFAULT_SUFFIX)


def _workspace(args: argparse.Namespace) -> RemoteWorkspace:
    """Собирает типизированное описание двух workspace-ов."""
    return RemoteWorkspace(
        local_root=Path.cwd().resolve(),
        host=args.host,
        remote_root=PurePosixPath(args.remote_root),
        remote_mlflow_port=args.remote_port,
        local_mlflow_port=args.local_port,
        remote_experiment=args.remote_experiment,
        destination_experiment=args.destination_experiment,
    )


def _sync_ids(workspace: RemoteWorkspace, args: argparse.Namespace) -> tuple[str, ...]:
    """Возвращает явные run ID или имена из выбранного этапа."""
    if args.run_id:
        return tuple(args.run_id)
    return selected_run_ids(workspace, args.series, args.stage, args.run_suffix)


def _sync_one(workspace: RemoteWorkspace, run_id: str, *, metrics_only: bool = False) -> None:
    """Копирует один run и печатает MLflow-связь."""
    result = sync_remote_run(workspace, run_id, include_checkpoints=not metrics_only)
    action = "copied" if result.copied else "verified"
    transfer = "imported" if result.mlflow.imported else "already imported"
    print(
        f"{run_id}: {action}; MLflow {transfer}; "
        f"local_run_id={result.mlflow.destination_run_id}; "
        f"metric_points={result.mlflow.metric_points}"
    )
    if metrics_only:
        print("Checkpoints: не проверены; для догрузки выполните sync без --metrics-only")


def _watch(workspace: RemoteWorkspace, args: argparse.Namespace) -> None:
    """Поочерёдно ждёт status=complete и импортирует run-ы."""
    run_ids = selected_run_ids(workspace, args.series, args.stage, args.run_suffix)
    for run_id in run_ids:
        while True:
            status = remote_run_status(workspace, run_id)
            if status == "complete":
                _sync_one(workspace, run_id)
                break
            if status == "failed":
                raise RuntimeError(f"Удалённый run завершился с ошибкой: {run_id}")
            print(f"{run_id}: ещё не завершён; следующая проверка через {args.poll_seconds} с")
            time.sleep(args.poll_seconds)


def _run_interactive(command: list[str]) -> int:
    """Запускает interactive-команду и тихо обрабатывает Ctrl+C."""
    try:
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 130


def main() -> int:
    """Выполняет одну операцию remote workflow."""
    args = parse_args()
    workspace = _workspace(args)
    if args.command == "prepare":
        print("Сверяю commit, хэши данных, GPU и uv-среду...")
        check = prepare_remote_workspace(workspace, install=not args.no_install)
        print(f"commit={check.commit}\nGPU: {check.gpu}")
        if args.preflight:
            print(preflight_remote(workspace, series_path=args.series, stage=args.stage))
        started = start_remote_mlflow(workspace)
        print(f"Remote MLflow: {'started' if started else 'already running'}")
    elif args.command == "start":
        started = start_remote_mlflow(workspace)
        run_ids = start_remote_training(
            workspace,
            series_path=args.series,
            stage=args.stage,
            suffix=args.run_suffix,
            session=args.session,
        )
        print(f"MLflow: {'started' if started else 'already running'}")
        print("Remote run IDs:", ", ".join(run_ids))
    elif args.command == "status":
        print(remote_status(workspace, args.session))
    elif args.command == "attach":
        return _run_interactive(["ssh", "-t", workspace.host, "zellij", "attach", args.session])
    elif args.command == "tunnel":
        print(f"Remote MLflow: http://127.0.0.1:{workspace.local_mlflow_port}")
        return _run_interactive(tunnel_command(workspace))
    elif args.command == "sync":
        for run_id in _sync_ids(workspace, args):
            _sync_one(workspace, run_id, metrics_only=args.metrics_only)
    elif args.command == "watch":
        _watch(workspace, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
