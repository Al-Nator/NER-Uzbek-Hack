"""Сохранение GP-кандидатов до NMS для дешёвых проверок без переобучения."""

import argparse
import time
from pathlib import Path

import torch

from uzner.data.io import load_documents, sha256_file
from uzner.domain import Document
from uzner.experiments.artifacts import write_json, write_jsonl
from uzner.training.frozen import FrozenPredictor
from uzner.training.span_prediction import SpanWindowScores, aggregate_span_candidates
from uzner.training.submission import checkpoint_hashes


def main() -> int:
    """Прогоняет frozen checkpoint на GPU без доступа модели к gold-разметке."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--floor", type=float, default=0.05)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.floor < 1:
        raise ValueError("floor должен лежать между 0 и 1")
    if not torch.cuda.is_available():
        raise RuntimeError("Полный модельный инференс требует GPU")
    gold = load_documents((("input", args.input),))
    documents = tuple(Document(d.hash, d.text) for d in gold)
    hashes = checkpoint_hashes(args.source_run)
    predictor = FrozenPredictor(args.source_run)
    started = time.perf_counter()
    records, windows, current = [], [], 0
    for window in predictor.windows(documents):
        while current < window.document_index:
            candidates = aggregate_span_candidates(windows, args.floor)
            records.append(
                {
                    "hash": documents[current].hash,
                    "entities": [e.to_mapping(include_score=True) for e in candidates],
                }
            )
            windows.clear()
            current += 1
            if current % 100 == 0:
                print(f"Кандидаты: {current}/{len(documents)}", flush=True)
        windows.append(SpanWindowScores(window.feature.offsets, window.probabilities.cpu()))
    while current < len(documents):
        candidates = aggregate_span_candidates(windows, args.floor)
        records.append(
            {
                "hash": documents[current].hash,
                "entities": [e.to_mapping(include_score=True) for e in candidates],
            }
        )
        windows.clear()
        current += 1
    write_jsonl(args.output, records)
    write_json(
        args.output.with_suffix(".manifest.json"),
        {
            "source_run": str(args.source_run),
            "checkpoint": "best",
            "floor": args.floor,
            "checkpoint_sha256": hashes,
            "input_sha256": sha256_file(args.input),
            "output_sha256": sha256_file(args.output),
            "documents": len(documents),
            "candidates": sum(len(r["entities"]) for r in records),
            "seconds": time.perf_counter() - started,
            "device": torch.cuda.get_device_name(),
            "bf16": True,
            "training": False,
            "gold_used_for_inference": False,
            "aggregation": "all covering windows, uniform mean, pre-NMS",
        },
    )
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
