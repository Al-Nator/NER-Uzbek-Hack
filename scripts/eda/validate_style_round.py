#!/usr/bin/env python3
"""Validate cumulative per-worker annotations after one <=50-document round."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from style_analysis_pipeline import (
    ANNOTATION_DIR,
    BATCH_DIR,
    GENRES,
    OUTPUT_FIELDS,
    REGISTERS,
    TEXT_FORMS,
    TOPICS,
    read_jsonl,
)


def validate(worker: str, part: int) -> dict[str, Any]:
    batch_paths = [
        BATCH_DIR / f"worker_{worker}_main_part{index:02d}.jsonl"
        for index in range(1, part + 1)
    ]
    errors: list[str] = []
    expected: dict[str, str] = {}
    for path in batch_paths:
        if not path.exists():
            errors.append(f"missing batch {path.name}")
            continue
        rows = read_jsonl(path)
        if len(rows) > 50:
            errors.append(f"{path.name} exceeds 50 documents")
        for row in rows:
            if row["item_id"] in expected:
                errors.append(f"duplicate expected item_id {row['item_id']}")
            expected[row["item_id"]] = row["text"]

    all_assigned: set[str] = set()
    for path in BATCH_DIR.glob(f"worker_{worker}_main_part*.jsonl"):
        all_assigned.update(row["item_id"] for row in read_jsonl(path))

    output_path = ANNOTATION_DIR / f"worker_{worker}.jsonl"
    if not output_path.exists():
        return {
            "worker": worker, "part": part, "valid": False,
            "expected_cumulative": len(expected), "received": 0,
            "errors": errors + [f"missing {output_path.name}"],
        }
    try:
        rows = read_jsonl(output_path)
    except ValueError as error:
        return {
            "worker": worker, "part": part, "valid": False,
            "expected_cumulative": len(expected), "received": 0,
            "errors": errors + [str(error)],
        }

    counts = Counter(row.get("item_id") for row in rows)
    duplicates = [item_id for item_id, count in counts.items() if count != 1]
    missing = sorted(set(expected) - set(counts))
    premature = sorted(set(counts) - set(expected))
    unassigned = sorted(set(counts) - all_assigned)
    if duplicates:
        errors.append(f"duplicate output item_ids ({len(duplicates)}): {duplicates[:10]}")
    if missing:
        errors.append(f"missing cumulative item_ids ({len(missing)}): {missing[:10]}")
    if premature:
        errors.append(f"unexpected/premature item_ids ({len(premature)}): {premature[:10]}")
    if unassigned:
        errors.append(f"unassigned item_ids ({len(unassigned)}): {unassigned[:10]}")

    for line_number, row in enumerate(rows, 1):
        prefix = f"line {line_number}"
        item_id = row.get("item_id")
        if set(row) != OUTPUT_FIELDS:
            errors.append(f"{prefix}: fields differ")
        if row.get("genre") not in GENRES:
            errors.append(f"{prefix}: invalid genre")
        topics = row.get("topics")
        if not isinstance(topics, list) or not 1 <= len(topics) <= 3:
            errors.append(f"{prefix}: topics must contain 1-3 labels")
        elif len(set(topics)) != len(topics) or any(topic not in TOPICS for topic in topics):
            errors.append(f"{prefix}: invalid/duplicate topics")
        if row.get("text_form") not in TEXT_FORMS:
            errors.append(f"{prefix}: invalid text_form")
        if row.get("register") not in REGISTERS:
            errors.append(f"{prefix}: invalid register")
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or len(evidence) > 3 or any(not isinstance(x, str) for x in evidence):
            errors.append(f"{prefix}: invalid evidence")
        elif item_id in expected:
            for excerpt in evidence:
                if excerpt not in expected[item_id]:
                    errors.append(f"{prefix}: evidence not exact substring: {excerpt!r}")
        for field in ("genre_confidence", "topics_confidence"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                errors.append(f"{prefix}: invalid {field}")

    return {
        "worker": worker,
        "part": part,
        "valid": not errors,
        "expected_cumulative": len(expected),
        "received": len(rows),
        "errors": errors[:100],
        "error_count": len(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", help="two-digit worker number; omit for all 20")
    parser.add_argument("--part", type=int, required=True)
    args = parser.parse_args()
    workers = [args.worker] if args.worker else [f"{number:02d}" for number in range(1, 21)]
    reports = [validate(worker, args.part) for worker in workers]
    payload = {"part": args.part, "all_valid": all(row["valid"] for row in reports), "workers": reports}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["all_valid"] else 1)


if __name__ == "__main__":
    main()
