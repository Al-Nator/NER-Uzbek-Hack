"""Полный GPU-прогон резидентного сервиса: качество, parity и профиль стадий."""

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from uzner.data.io import load_documents, read_jsonl, sha256_file, write_predictions
from uzner.domain import Document, Prediction
from uzner.evaluation.metrics import evaluate_predictions
from uzner.experiments.artifacts import write_json
from uzner.serving.config import RuntimeConfig
from uzner.serving.measurement import compare_predictions, hardware, latency_summary
from uzner.serving.runtime import ResidentEnsemble


def main() -> None:
    """Отделяет warmup/загрузку от steady-state и всегда сохраняет исходные ответы."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gold", action="store_true")
    args = parser.parse_args()
    if args.batch < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch и limit должны быть положительными")
    args.output.mkdir(parents=True, exist_ok=False)
    gold = load_documents((("input", args.input),))
    if args.limit:
        gold = gold[: args.limit]
    documents = tuple(Document(d.hash, d.text, ()) for d in gold)
    config = RuntimeConfig.read(args.config)
    before = perf_counter()
    runtime = ResidentEnsemble(config)
    load_seconds = perf_counter() - before
    for _ in range(3):
        runtime.predict_documents(documents[: args.batch])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings, predictions, stages = [], [], defaultdict(float)
    started = perf_counter()
    for start in range(0, len(documents), args.batch):
        before = perf_counter()
        predictions.extend(runtime.predict_documents(documents[start : start + args.batch]))
        timings.append(perf_counter() - before)
        for key, value in runtime.timings.items():
            stages[key] += value
        if start % 256 == 0:
            print(f"{start}/{len(documents)} {perf_counter() - started:.1f}s", flush=True)
    seconds = perf_counter() - started
    write_predictions(args.output / "predictions.jsonl", predictions)
    summary = {
        "kind": "in_process_gpu_profile",
        "pilot": bool(args.limit),
        "hardware": hardware(),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "input_sha256": sha256_file(args.input),
        "documents": len(documents),
        "request_batch": args.batch,
        "warmup_requests": 3,
        "load_seconds": load_seconds,
        "seconds": seconds,
        "documents_per_second": len(documents) / seconds,
        "latency": latency_summary(timings),
        "stages_seconds": dict(stages),
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "engines": {c.name: c.engine is not None for c in runtime.components},
    }
    if args.gold:
        summary["quality"] = evaluate_predictions(gold, predictions).to_mapping()
    if args.reference:
        keys = {d.hash for d in documents}
        reference = tuple(
            Prediction.from_mapping(row)
            for row in read_jsonl(args.reference)
            if row["hash"] in keys
        )
        parity = compare_predictions(reference, predictions)
        write_json(args.output / "parity.json", parity)
        summary["parity"] = {key: value for key, value in parity.items() if key != "changes"}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "hardware"}), flush=True)


if __name__ == "__main__":
    main()
