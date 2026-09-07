"""Возвращает шесть A100-run-ов серии 7 вместе с весами и MLflow-историей."""

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import yaml

from uzner.experiments.remote import RemoteWorkspace, ssh_command
from uzner.experiments.remote_sync import sync_remote_run
from uzner.training.reranker_cache import PROPOSERS


def main() -> None:
    """По готовности копирует каждый run; повторный импорт идемпотентен."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Одна проверка без ожидания")
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/seventh/rerankers.yaml").read_text())
    pending = [*PROPOSERS, *(v["run_id"] for v in config["variants"])]
    workspace = RemoteWorkspace(local_root=Path.cwd())
    while pending:
        for name in pending[:]:
            result = subprocess.run(
                ssh_command(
                    workspace, ["cat", str(workspace.remote_root / "runs" / name / "status.json")]
                ),
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            if result.returncode:
                continue
            state = json.loads(result.stdout)
            if state.get("status") == "failed":
                raise RuntimeError(f"Ошибка удалённого run: {name}")
            expected = "reranker_proposer" if name in PROPOSERS else "span_reranker"
            if state.get("status") != "complete" or state.get("kind") != expected:
                continue
            if name not in PROPOSERS:
                subprocess.run(
                    [
                        "rsync",
                        "--archive",
                        "--partial",
                        f"{workspace.host}:{workspace.remote_root}/{config['split_root']}/",
                        f"{config['split_root']}/",
                    ],
                    check=True,
                )
            destination = "uzner-seventh-proposers" if name in PROPOSERS else "uzner-seventh-series"
            imported = sync_remote_run(replace(workspace, destination_experiment=destination), name)
            print(f"Вернулся {name}: MLflow {imported.mlflow.destination_run_id}", flush=True)
            pending.remove(name)
        if args.once:
            break
        if pending:
            print(f"Ожидаю: {', '.join(pending)}", flush=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
