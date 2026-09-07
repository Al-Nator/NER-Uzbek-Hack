"""Последовательная A100-очередь без загрузки моделей в родительский процесс."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from uzner.config import load_experiment_config


def main() -> None:
    """Запускает три proposer-а, продолжая только подтверждённые завершённые runs."""
    for config in sorted(Path("configs/seventh").glob("s7p*.yaml")):
        run_id = load_experiment_config(config).run_id
        status = Path("runs") / run_id / "status.json"
        if status.exists():
            if (
                json.loads(status.read_text()).get("kind") == "reranker_proposer"
                and json.loads(status.read_text()).get("status") == "complete"
            ):
                continue
            raise FileExistsError(f"Незавершённый run требует явного resume: {run_id}")
        if shutil.disk_usage(".").free < 25 * 1024**3:
            raise RuntimeError("Для безопасного checkpoint недостаточно 25 GiB свободного места")
        subprocess.run(
            [sys.executable, "-u", "scripts/train_reranker_proposers.py", "--config", str(config)],
            check=True,
        )


if __name__ == "__main__":
    main()
