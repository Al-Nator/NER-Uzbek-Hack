"""CLI последовательных этапов конфигурируемой серии."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uzner.config import load_experiment_config, with_run_suffix
from uzner.experiments.series import load_series
from uzner.training.engine import TrainRequest
from uzner.training.runner import run_experiment


def parse_args() -> argparse.Namespace:
    """Разбирает этап серии и безопасный пропуск готовых run-ов."""
    parser = argparse.ArgumentParser(description="Запустить этап серии экспериментов")
    parser.add_argument("--series", type=Path, default=Path("configs/series/first.yaml"))
    parser.add_argument("--stage", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--run-suffix")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _is_complete(project_root: Path, run_id: str) -> bool:
    """Проверяет status без доверия к наличию каталога."""
    path = project_root / "runs" / run_id / "status.json"
    if not path.is_file():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("status") == "complete"


def main() -> int:
    """Выполняет run-ы строго последовательно."""
    args = parse_args()
    project_root = args.project_root.resolve()
    series = load_series((project_root / args.series).resolve())
    for config_name in series.select(args.stage):
        config_path = (project_root / config_name).resolve()
        config = with_run_suffix(load_experiment_config(config_path), args.run_suffix)
        run_id = config.run_id
        if args.dry_run:
            print(f"{run_id}: {config_path}")
            continue
        if args.skip_complete and _is_complete(project_root, run_id):
            print(f"skip complete: {run_id}")
            continue
        run_experiment(
            TrainRequest(
                config_path=config_path,
                project_root=project_root,
                run_id_suffix=args.run_suffix,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
