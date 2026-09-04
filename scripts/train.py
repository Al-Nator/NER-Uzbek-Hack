"""CLI одного воспроизводимого NER-эксперимента."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uzner.training.engine import TrainRequest
from uzner.training.runner import run_experiment


def parse_args() -> argparse.Namespace:
    """Разбирает полный и изолированный smoke-режимы."""
    parser = argparse.ArgumentParser(description="Запустить один NER-эксперимент")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-suffix")
    parser.add_argument("--max-train-documents", type=int)
    parser.add_argument("--max-dev-documents", type=int)
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Число эпох только для изолированного smoke-run",
    )
    parser.add_argument(
        "--smoke-output",
        type=Path,
        help="Отдельный output root; такой run не попадёт в reports/experiments.csv",
    )
    return parser.parse_args()


def main() -> int:
    """Запускает experiment и печатает компактный машинный итог."""
    args = parse_args()
    isolated = args.smoke_output is not None
    request = TrainRequest(
        config_path=args.config,
        project_root=args.project_root,
        resume=args.resume,
        max_train_documents=args.max_train_documents,
        max_dev_documents=args.max_dev_documents,
        max_epochs=args.max_epochs,
        publish_summary=not isolated,
        output_root_override=args.smoke_output,
        run_id_suffix=args.run_suffix,
    )
    result = run_experiment(request)
    print(
        json.dumps(
            {
                "status": "complete",
                "run_path": str(result.paths.root),
                "best_epoch": result.best_epoch,
                "best_micro_f1": result.best_micro_f1,
                "published": request.publish_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
