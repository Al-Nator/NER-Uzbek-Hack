"""Удаляет локальные checkpoint-ы завершённых слабых runs, сохраняя отчёты."""

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from uzner.experiments.artifacts import write_json


@dataclass(frozen=True)
class Candidate:
    """Проверенная цель очистки с метрикой завершённого запуска."""

    path: str
    score: float


def candidates(root: Path, protected: set[str]) -> list[Candidate]:
    """Выбирает только обычные каталоги завершённых runs с известной метрикой."""
    result = []
    for run in sorted(root.iterdir()):
        status = run / "status.json"
        checkpoint = run / "checkpoints"
        if run.name in protected or run.is_symlink() or not status.is_file():
            continue
        if not checkpoint.is_dir() or checkpoint.is_symlink():
            continue
        data = json.loads(status.read_text())
        score = data.get("best_micro_f1")
        if (
            data.get("status") == "complete"
            and isinstance(score, int | float)
            and 0 <= score < 0.9
        ):
            result.append(Candidate(str(checkpoint.resolve()), score))
    return result


def main() -> None:
    """Пишет план, затем удаляет только явно выбранные checkpoint-каталоги."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    protected = set(
        json.loads(
            (root / "runs/s62_span_majority_full-targeted-v1/resolved_config.json").read_text()
        )["sources"]
    )
    roots = [root / "runs", root / "artifacts/remote_archive/alnator_20260906_v1/runs"]
    selected = [c for directory in roots for c in candidates(directory, protected)]
    report = root / "artifacts/cleanup/low_checkpoint_20260907.json"
    write_json(
        report,
        {
            "targets": [asdict(c) for c in selected],
            "protected": sorted(protected),
            "applied": False,
        },
    )
    for candidate in selected:
        print(candidate.path, candidate.score, flush=True)
        if args.apply:
            path = Path(candidate.path)
            if path.name != "checkpoints" or path.parent.parent not in roots:
                raise ValueError("Цель вне разрешённых каталогов")
            shutil.rmtree(path)
            write_json(
                path.parent / "checkpoints_removed.json",
                {
                    "reason": "user requested best exact micro-F1 < 0.9 cleanup",
                    "best_micro_f1": candidate.score,
                    "metrics_and_predictions_preserved": True,
                },
            )
    if args.apply:
        write_json(
            report,
            {
                "targets": [asdict(c) for c in selected],
                "protected": sorted(protected),
                "applied": True,
            },
        )


if __name__ == "__main__":
    main()
