from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LABELS = ("ORG", "NAME", "GEO")
JsonObject = dict[str, Any]
EntityKey = tuple[str, int, int]


def parse_args() -> argparse.Namespace:
    """Разбирает пути gold, предсказаний и необязательного JSON-отчёта."""

    parser = argparse.ArgumentParser(description="Calculate exact-span NER metrics.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path, kind: str) -> list[JsonObject]:
    """Читает непустой JSONL и проверяет уникальность hash."""

    records: list[JsonObject] = []
    seen_hashes: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: empty line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            record_hash = record.get("hash")
            if not isinstance(record_hash, str) or not record_hash:
                raise ValueError(f"{path}:{line_number}: hash must be a non-empty string")
            if record_hash in seen_hashes:
                raise ValueError(f"{path}:{line_number}: duplicate hash {record_hash}")
            seen_hashes.add(record_hash)
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no {kind} records")
    return records


def _validate_entities(raw: Any, text_length: int, source: str) -> set[EntityKey]:
    """Проверяет сущности записи и возвращает множество exact-span ключей."""

    if not isinstance(raw, list):
        raise ValueError(f"{source}: entities must be an array")
    entities: set[EntityKey] = set()
    for index, entity in enumerate(raw):
        if not isinstance(entity, dict):
            raise ValueError(f"{source}/entities[{index}]: entity must be an object")
        label = entity.get("label")
        start = entity.get("start")
        end = entity.get("end")
        if label not in LABELS:
            raise ValueError(f"{source}/entities[{index}]: invalid label {label!r}")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= text_length
        ):
            raise ValueError(f"{source}/entities[{index}]: invalid offsets")
        key = (label, start, end)
        if key in entities:
            raise ValueError(f"{source}/entities[{index}]: duplicate entity")
        entities.add(key)
    return entities


def _gold_by_hash(records: list[JsonObject], path: Path) -> dict[str, JsonObject]:
    """Проверяет gold-тексты и индексирует их по hash."""

    result: dict[str, JsonObject] = {}
    for index, record in enumerate(records, start=1):
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{path}:{index}: gold text must be a string")
        entities = _validate_entities(record.get("entities"), len(text), f"{path}:{index}")
        result[record["hash"]] = {"text": text, "entities": entities}
    return result


def _predictions_by_hash(
    records: list[JsonObject],
    path: Path,
    gold: dict[str, JsonObject],
) -> dict[str, set[EntityKey]]:
    """Проверяет соответствие hash и координат предсказаний gold-текстам."""

    predicted_hashes = {record["hash"] for record in records}
    gold_hashes = set(gold)
    missing = sorted(gold_hashes - predicted_hashes)
    extra = sorted(predicted_hashes - gold_hashes)
    if missing or extra:
        raise ValueError(
            "gold/prediction hashes differ: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"extra={extra[:5]} ({len(extra)} total)"
        )

    result: dict[str, set[EntityKey]] = {}
    for index, record in enumerate(records, start=1):
        record_hash = record["hash"]
        gold_record = gold[record_hash]
        if "text" in record and record["text"] != gold_record["text"]:
            raise ValueError(f"{path}:{index}: prediction text differs from gold for {record_hash}")
        result[record_hash] = _validate_entities(
            record.get("entities"),
            len(gold_record["text"]),
            f"{path}:{index}",
        )
    return result


def _metric_values(tp: int, fp: int, fn: int) -> JsonObject:
    """Вычисляет Precision, Recall и F1 из TP, FP и FN."""

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gold": tp + fn,
        "predicted": tp + fp,
    }


def calculate_metrics(
    gold: dict[str, JsonObject],
    predictions: dict[str, set[EntityKey]],
) -> JsonObject:
    """Считает exact-span метрики по классам, micro и macro."""

    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
    for record_hash, gold_record in gold.items():
        gold_entities = gold_record["entities"]
        predicted_entities = predictions[record_hash]
        for label in LABELS:
            gold_label = {entity for entity in gold_entities if entity[0] == label}
            predicted_label = {entity for entity in predicted_entities if entity[0] == label}
            counts[label]["tp"] += len(gold_label & predicted_label)
            counts[label]["fp"] += len(predicted_label - gold_label)
            counts[label]["fn"] += len(gold_label - predicted_label)

    by_label = {
        label: _metric_values(values["tp"], values["fp"], values["fn"])
        for label, values in counts.items()
    }
    micro = _metric_values(
        sum(values["tp"] for values in counts.values()),
        sum(values["fp"] for values in counts.values()),
        sum(values["fn"] for values in counts.values()),
    )
    macro = {
        metric: sum(by_label[label][metric] for label in LABELS) / len(LABELS)
        for metric in ("precision", "recall", "f1")
    }
    return {
        "schema_version": 1,
        "matching": "same hash and exact label/start/end",
        "records": len(gold),
        "by_label": by_label,
        "micro": micro,
        "macro": macro,
    }


def print_metrics(metrics: JsonObject) -> None:
    """Печатает компактную таблицу основных метрик."""

    header = (
        f"{'scope':<8} {'precision':>10} {'recall':>10} {'f1':>10} {'tp':>8} {'fp':>8} {'fn':>8}"
    )
    print(header)
    print("-" * len(header))
    for label in LABELS:
        values = metrics["by_label"][label]
        print(
            f"{label:<8} {values['precision']:>10.4f} {values['recall']:>10.4f} "
            f"{values['f1']:>10.4f} {values['tp']:>8} {values['fp']:>8} {values['fn']:>8}"
        )
    micro = metrics["micro"]
    print(
        f"{'micro':<8} {micro['precision']:>10.4f} {micro['recall']:>10.4f} "
        f"{micro['f1']:>10.4f} {micro['tp']:>8} {micro['fp']:>8} {micro['fn']:>8}"
    )
    macro = metrics["macro"]
    print(
        f"{'macro':<8} {macro['precision']:>10.4f} {macro['recall']:>10.4f} "
        f"{macro['f1']:>10.4f} {'-':>8} {'-':>8} {'-':>8}"
    )


def load_gold(path: Path) -> tuple[list[JsonObject], dict[str, JsonObject]]:
    """Читает и проверяет gold, возвращая записи и индекс по hash."""

    records = read_jsonl(path, "gold")
    return records, _gold_by_hash(records, path)


def evaluate_files(gold_path: Path, predictions_path: Path) -> JsonObject:
    """Валидирует два JSONL-файла и рассчитывает exact-span метрики."""

    _, gold = load_gold(gold_path)
    predictions = _predictions_by_hash(
        read_jsonl(predictions_path, "prediction"),
        predictions_path,
        gold,
    )
    return calculate_metrics(gold, predictions)


def write_metrics(path: Path, metrics: JsonObject) -> None:
    """Записывает JSON-отчёт с метриками."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> JsonObject:
    """Валидирует файлы, рассчитывает метрики и при необходимости пишет JSON."""

    gold_path = args.gold.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    metrics = evaluate_files(gold_path, predictions_path)
    print_metrics(metrics)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        write_metrics(output_path, metrics)
        print(f"Metrics: {output_path}")
    return metrics


def main() -> int:
    """Запускает scorer с компактным сообщением об ошибке."""

    try:
        run(parse_args())
    except (OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
