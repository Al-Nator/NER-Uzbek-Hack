"""Запускает новую GPU-очередь после завершения явно указанных процессов."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from uzner.config import load_experiment_config, with_run_suffix
from uzner.experiments.artifacts import write_json
from uzner.experiments.deferred import ProcessIdentity
from uzner.experiments.series import load_series


def main() -> int:
    """Ждёт зависимости и сохраняет состояние очереди; независимые сбои не скрывает."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for", action="append", required=True)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--run-suffix", required=True)
    parser.add_argument("--queue-state", type=Path, required=True)
    args = parser.parse_args()
    dependencies = [ProcessIdentity.parse(value) for value in args.wait_for]
    series = load_series(args.series)
    if args.queue_state.exists():
        raise FileExistsError(args.queue_state)
    state = {
        "status": "waiting",
        "dependencies": args.wait_for,
        "series": str(args.series),
        "suffix": args.run_suffix,
        "results": {},
    }
    write_json(args.queue_state, state)
    print(json.dumps(state), flush=True)
    while any(process.alive() for process in dependencies):
        time.sleep(30)
    for path in series.select("all"):
        config = with_run_suffix(load_experiment_config(Path(path)), args.run_suffix)
        state.update(status="running", current=config.run_id)
        write_json(args.queue_state, state)
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/train.py",
                "--config",
                path,
                "--run-suffix",
                args.run_suffix,
            ],
            check=False,
        )
        state["results"][config.run_id] = result.returncode
        write_json(args.queue_state, state)
    failed = any(state["results"].values())
    state["status"] = "failed" if failed else "complete"
    write_json(args.queue_state, state)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
