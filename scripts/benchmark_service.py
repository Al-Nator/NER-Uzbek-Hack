"""End-to-end HTTP benchmark с cold/warm разграничением и полным span parity."""

import argparse
import json
import math
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter, sleep

import httpx

from uzner.data.io import load_documents, read_jsonl, sha256_file, write_predictions
from uzner.domain import Document, Prediction
from uzner.evaluation.metrics import evaluate_predictions
from uzner.experiments.artifacts import write_json
from uzner.serving.http_client import HttpSettings, ResponseSample, request, wait_ready
from uzner.serving.measurement import compare_predictions, hardware, latency_summary
from uzner.serving.memory import MemorySampler


def request_scheduled(
    client: httpx.Client, documents: tuple[Document, ...], scheduled: float
) -> ResponseSample:
    """Для open-loop учитывает ожидание от заданного времени прихода запроса."""
    sleep(max(0.0, scheduled - perf_counter()))
    lag = max(0.0, perf_counter() - scheduled)
    result = request(client, documents)
    return ResponseSample(perf_counter() - scheduled, result.predictions, lag)


def main(argv: list[str] | None = None) -> None:
    """Измеряет реальные сообщения, а не подготовленные тензоры или кэш ответов."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=[1, 8], default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--arrival-rate", type=float)
    parser.add_argument("--gold", action="store_true")
    parser.add_argument("--gpu-memory", choices=("auto", "required", "off"), default="auto")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    args = parser.parse_args(argv)
    if min(args.concurrency, args.rounds) < 1:
        parser.error("rounds/concurrency должны быть положительными")
    if args.arrival_rate is not None and (
        not math.isfinite(args.arrival_rate) or args.arrival_rate <= 0
    ):
        parser.error("arrival-rate должен быть положительным")
    settings = HttpSettings(args.url, args.timeout, args.startup_timeout)
    runtime_config = json.loads(args.config.read_text("utf-8"))
    gold = load_documents((("benchmark", args.input),), require_entities=args.gold)
    if not gold:
        raise ValueError("Benchmark input не должен быть пустым")
    reference = None
    if args.reference:
        rows = read_jsonl(args.reference)
        if any(not isinstance(row.get("entities"), list) for row in rows):
            raise ValueError("Reference должен содержать entities[] в каждой строке")
        reference = tuple(Prediction.from_mapping(row) for row in rows)
        evaluate_predictions(gold, reference)
    args.output.mkdir(parents=True, exist_ok=False)
    documents = tuple(Document(d.hash, d.text, ()) for d in gold)
    batches = tuple(documents[i : i + args.batch] for i in range(0, len(documents), args.batch))
    samples, rounds, dispatch_lags = [], [], []
    with httpx.Client(
        base_url=settings.url,
        timeout=settings.timeout,
        limits=httpx.Limits(max_connections=args.concurrency),
        trust_env=False,
    ) as client:
        wait_ready(client, settings.startup_timeout)
        for _ in range(3):
            request(client, batches[0])
        memory = MemorySampler(args.gpu_memory)
        memory.start()
        started = perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                for iteration in range(args.rounds):
                    if args.arrival_rate:
                        jobs = [
                            executor.submit(
                                request_scheduled,
                                client,
                                group,
                                started + (iteration * len(batches) + index) / args.arrival_rate,
                            )
                            for index, group in enumerate(batches)
                        ]
                        values = [job.result() for job in jobs]
                    else:
                        values = list(executor.map(lambda group: request(client, group), batches))
                    rounds.append(tuple(p for value in values for p in value.predictions))
                    samples.extend(value.seconds for value in values)
                    dispatch_lags.extend(value.dispatch_lag_seconds for value in values)
                    print(f"round {iteration + 1}/{args.rounds}", flush=True)
            elapsed = perf_counter() - started
        finally:
            memory_result = memory.finish()
    lengths = sorted(len(doc.text) for doc in documents)
    summary = {
        "kind": "end_to_end_http",
        "load_mode": "open_loop" if args.arrival_rate else "closed_loop",
        "arrival_rate_requests_per_second": args.arrival_rate,
        "url": args.url,
        "hardware": hardware(),
        "config": runtime_config,
        "config_identity": "Client-supplied expected runtime; not verified by server handshake",
        "config_sha256": sha256_file(args.config),
        "input_sha256": sha256_file(args.input),
        "unique_documents": len(documents),
        "text_characters": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "mean": statistics.mean(lengths),
            "p95": lengths[math.ceil(len(lengths) * 0.95) - 1],
            "max": max(lengths),
        },
        "messages": len(documents) * args.rounds,
        "requests": len(samples),
        "batch": args.batch,
        "concurrency": args.concurrency,
        "rounds": args.rounds,
        "warmup_requests": 3,
        "includes": [
            "HTTP",
            "validation",
            "tokenization",
            "all 3 encoders",
            "window merge",
            "decoding",
            "dictionary",
            "repeats",
            "JSON",
        ],
        "excludes": ["model loading", "warmup"],
        "client_validation": "Included in throughput wall time; latency ends after JSON parsing.",
        "http_errors": 0,
        "response_cache": False,
        "elapsed_seconds": elapsed,
        "messages_per_second": len(documents) * args.rounds / elapsed,
        "requests_per_second": len(samples) / elapsed,
        "latency": latency_summary(samples),
        "client_dispatch_lag": latency_summary(dispatch_lags),
        "gpu_memory": memory_result,
    }
    write_predictions(args.output / "predictions.jsonl", rounds[0])
    write_json(args.output / "latencies.json", samples)
    for index, predictions in enumerate(rounds[1:], 1):
        parity = compare_predictions(rounds[0], predictions)
        write_json(args.output / f"repeat_parity_{index}.json", parity)
        if parity["changed_documents"]:
            raise ValueError("Повторный HTTP-прогон изменил spans")
    if args.gold:
        summary["quality"] = evaluate_predictions(gold, rounds[0]).to_mapping()
    if args.reference:
        parity = compare_predictions(reference, rounds[0])
        write_json(args.output / "parity.json", parity)
        summary["parity"] = {k: v for k, v in parity.items() if k != "changes"}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "hardware"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, TypeError, OSError, RuntimeError, httpx.HTTPError) as error:
        print(f"benchmark: {error}", file=sys.stderr)
        raise SystemExit(2) from error
