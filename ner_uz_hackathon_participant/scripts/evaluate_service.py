from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from check_service import (
    ContractError,
    normalize_base_url,
    predict_batch,
    validate_timeouts,
    wait_for_health,
)
from evaluate import evaluate_files, load_gold, print_metrics, write_metrics

JsonObject = dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Разбирает адрес сервиса, пути и параметры батчевого инференса."""

    parser = argparse.ArgumentParser(
        description="Run a service on gold JSONL and calculate exact-span NER metrics."
    )
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--gold", type=Path, default=Path("data/dev.jsonl"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/service/dev_predictions.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/service/dev_metrics.json"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Проверяет пути, исключая перезапись gold или predictions метриками."""

    gold_path = args.gold.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if gold_path == predictions_path:
        raise ValueError("predictions path must differ from gold path")
    if output_path in {gold_path, predictions_path}:
        raise ValueError("output path must differ from gold and predictions paths")
    return gold_path, predictions_path, output_path


def _compact_prediction(result: JsonObject) -> JsonObject:
    """Оставляет только поля, используемые exact-span scorer."""

    entities = [
        {
            "label": entity["label"],
            "start": entity["start"],
            "end": entity["end"],
        }
        for entity in result["entities"]
    ]
    return {"hash": result["hash"], "entities": entities}


def _write_jsonl(path: Path, records: list[JsonObject]) -> None:
    """Записывает проверенные ответы сервиса в JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def run(args: argparse.Namespace) -> JsonObject:
    """Прогоняет gold через HTTP API, сохраняет ответы и считает метрики."""

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    validate_timeouts(args.startup_timeout, args.request_timeout)
    base_url = normalize_base_url(args.url)
    gold_path, predictions_path, output_path = _resolve_paths(args)
    gold_records, _ = load_gold(gold_path)

    wait_for_health(base_url, args.startup_timeout, args.request_timeout)
    print("OK  GET /healthz")

    predictions: list[JsonObject] = []
    entity_count = 0
    batch_count = math.ceil(len(gold_records) / args.batch_size)
    for batch_index, start in enumerate(
        range(0, len(gold_records), args.batch_size),
        start=1,
    ):
        records = gold_records[start : start + args.batch_size]
        inputs = [{"hash": record["hash"], "text": record["text"]} for record in records]
        try:
            results, batch_entity_count = predict_batch(
                base_url,
                inputs,
                args.request_timeout,
            )
        except ContractError as error:
            raise ContractError(
                f"batch {batch_index}/{batch_count}, records {start + 1}-{start + len(records)}: "
                f"{error}"
            ) from error
        predictions.extend(_compact_prediction(result) for result in results)
        entity_count += batch_entity_count
        if batch_index % 10 == 0 or batch_index == batch_count:
            print(
                f"Predict: {start + len(records)}/{len(gold_records)} records "
                f"({batch_index}/{batch_count} batches)"
            )

    _write_jsonl(predictions_path, predictions)
    print(f"Predictions: {predictions_path} ({entity_count} entities)")

    metrics = evaluate_files(gold_path, predictions_path)
    print_metrics(metrics)
    write_metrics(output_path, metrics)
    print(f"Metrics: {output_path}")
    return metrics


def main() -> int:
    """Запускает service evaluation с компактным сообщением об ошибке."""

    try:
        run(parse_args())
    except (ContractError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
