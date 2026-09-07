"""Запускает s62 train+dev после архива и возвращает inference-веса с MLflow."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from uzner.data.io import sha256_file
from uzner.experiments.artifacts import write_json
from uzner.experiments.remote import RemoteWorkspace, ssh_command, start_remote_mlflow
from uzner.experiments.remote_sync import remote_run_status, sync_remote_run
from uzner.training.final_ensemble import ENSEMBLE_ID, STAGES

ARCHIVE = Path("artifacts/remote_archive/alnator_20260906_v1")
STATE = Path("artifacts/final_ensemble_queue_v1/status.json")
SESSION = "uzner-s62-train-dev-v1"


def wait_for_archive() -> None:
    """Не позволяет смешать новые runs со снимком старых перед очисткой."""
    while not (ARCHIVE / "deletion_complete.json").exists():
        write_json(STATE, {"status": "waiting_for_verified_archive", "session": SESSION})
        active = (
            subprocess.run(
                ["systemctl", "--user", "is-active", "--quiet", "uzner-archive-a100.service"],
                check=False,
            ).returncode
            == 0
        )
        if not active:
            raise RuntimeError(
                "Архивирование не завершено, служба остановлена; обучение не запущено"
            )
        print("Ожидаю две проверки и очистку старых A100 runs", flush=True)
        time.sleep(60)
    report = json.loads((ARCHIVE / "deletion_complete.json").read_text())
    if report.get("verification_passes") != 2 or not report.get("removed_remote_runs"):
        raise ValueError("Нет подтверждения безопасной очистки")


def start(workspace: RemoteWorkspace) -> None:
    """Создаёт самостоятельную GPU-очередь в zellij с отдельным MLflow experiment."""
    start_remote_mlflow(workspace)
    for command in (
        ["zellij", "attach", "--create-background", SESSION],
        [
            "zellij",
            "--session",
            SESSION,
            "run",
            "--cwd",
            str(workspace.remote_root),
            "--name",
            "s62-train-dev",
            "--",
            "env",
            "UZNER_MLFLOW_EXPERIMENT=uzner-final-fit-a100",
            "uv",
            "run",
            "python",
            "scripts/run_final_ensemble.py",
        ],
    ):
        subprocess.run(ssh_command(workspace, command), check=True)
    write_json(
        STATE, {"status": "training", "session": SESSION, "stages": [s.run_id for s in STAGES]}
    )


def return_inference_weights(workspace: RemoteWorkspace, run_id: str) -> None:
    """Возвращает best inference-веса; optimizer/last остаются pending из-за места."""
    run = Path("runs") / run_id
    destination = run / "checkpoints/best"
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-aH",
            "--partial",
            "--exclude=training_state.pt",
            f"{workspace.host}:{workspace.remote_root}/runs/{run_id}/checkpoints/best/",
            f"{destination}/",
        ],
        check=True,
    )
    manifest = json.loads((run / "artifact_manifest.json").read_text())
    verified = []
    for entry in manifest["artifacts"]:
        name = entry["path"]
        if not name.startswith("checkpoints/best/") or name.endswith("training_state.pt"):
            continue
        path = run / name
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Повреждён checkpoint: {name}")
        verified.append(name)
    if not any(name.endswith("head.safetensors") for name in verified):
        raise ValueError("В манифесте отсутствует голова модели")
    write_json(
        run / "inference_weights_verified.json",
        {
            "files": verified,
            "full_resume_checkpoint_local": False,
            "optimizer_and_last": "preserved on A100; transfer pending",
        },
    )


def main() -> None:
    """Оркестрирует только согласованный рецепт, не отправляя ничего на лидерборд."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-ready", type=Path)
    parser.add_argument("--attach-existing", action="store_true")
    args = parser.parse_args()
    if args.attach_existing:
        pass
    elif args.space_ready:
        report = json.loads(args.space_ready.read_text())
        if report.get("verification_passes") != 2 or not report.get("removed_remote_runs"):
            raise ValueError("Нет двойной проверки освобождённого места")
    else:
        wait_for_archive()
    workspace = RemoteWorkspace(local_root=Path.cwd(), remote_experiment="uzner-final-fit-a100")
    if not args.attach_existing:
        start(workspace)
    else:
        write_json(STATE, {"status": "training", "session": SESSION})
    for stage in STAGES:
        while True:
            status = remote_run_status(workspace, stage.run_id)
            if status == "complete":
                break
            if status == "failed":
                raise RuntimeError(f"Обучение завершилось ошибкой: {stage.run_id}")
            print(f"Ожидаю {stage.run_id}: {status}", flush=True)
            time.sleep(60)
        result = sync_remote_run(workspace, stage.run_id, include_checkpoints=False)
        if stage.ensemble_member:
            return_inference_weights(workspace, stage.run_id)
        print(f"Returned {stage.run_id}; MLflow={result.mlflow.destination_run_id}", flush=True)
    # Описание ансамбля появляется после завершения всех трёх компонентов.
    subprocess.run(
        [
            "rsync",
            "-a",
            f"{workspace.host}:{workspace.remote_root}/runs/{ENSEMBLE_ID}/",
            f"runs/{ENSEMBLE_ID}/",
        ],
        check=True,
    )
    write_json(
        STATE,
        {
            "status": "complete",
            "ensemble_run": ENSEMBLE_ID,
            "inference_weights_local": True,
            "full_resume_transfer_pending": True,
        },
    )


if __name__ == "__main__":
    main()
