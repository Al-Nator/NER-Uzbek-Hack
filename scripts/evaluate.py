"""Расчёт новых exact-span метрик из командной строки."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from uzner.evaluation.files import evaluate_files


def parse_args() -> argparse.Namespace:
    """Разбирает пути gold, predictions и необязательного отчёта."""
    parser = argparse.ArgumentParser(description="Посчитать exact-span NER метрики")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Загружает файлы, считает метрики и печатает JSON."""
    args = parse_args()
    try:
        report = evaluate_files(args.gold, args.predictions, args.output).to_mapping()
    except (ValueError, TypeError, OSError) as error:
        print(f"eval: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
