"""Расчёт новых exact-span метрик из командной строки."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uzner.data.io import read_jsonl
from uzner.domain import Document, Prediction
from uzner.evaluation.metrics import evaluate_predictions


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
    gold = [
        Document.from_mapping(record, source=args.gold.name) for record in read_jsonl(args.gold)
    ]
    predictions = [Prediction.from_mapping(record) for record in read_jsonl(args.predictions)]
    report = evaluate_predictions(gold, predictions).to_mapping()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
