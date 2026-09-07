"""Сравнение изменений и парная неопределённость на документах dev."""

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from uzner.domain import Document, Prediction
from uzner.evaluation.metrics import evaluate_predictions
from uzner.experiments.artifacts import write_json, write_jsonl
from uzner.posthoc.selection import entity_key


@dataclass(frozen=True)
class Changes:
    """Число исправлений и новых ошибок относительно зафиксированной системы."""

    added_tp: int
    added_fp: int
    removed_tp: int
    removed_fp: int
    changed_documents: int


def save_changes(
    root: Path,
    gold: tuple[Document, ...],
    baseline: tuple[Prediction, ...],
    predictions: tuple[Prediction, ...],
) -> Changes:
    """Сохраняет каждый изменённый span с контекстом и точной gold-проверкой."""
    records, totals, changed = [], [0, 0, 0, 0], 0
    for document, old, new in zip(gold, baseline, predictions, strict=True):
        expected = {entity_key(e) for e in document.entities}
        before, after = {entity_key(e) for e in old.entities}, {entity_key(e) for e in new.entities}
        changed += before != after
        for operation, keys, offset in (
            ("added", after - before, 0),
            ("removed", before - after, 2),
        ):
            for start, end, label in sorted(keys):
                correct = (start, end, label) in expected
                totals[offset + (not correct)] += 1
                records.append(
                    {
                        "hash": document.hash,
                        "operation": operation,
                        "label": label,
                        "start": start,
                        "end": end,
                        "correct": correct,
                        "surface": document.text[start:end],
                        "context": document.text[max(0, start - 60) : end + 60],
                    }
                )
    result = Changes(*totals, changed)
    write_jsonl(root / "changes.jsonl", records)
    write_json(root / "changes.json", asdict(result))
    return result


def split_metrics(
    gold: tuple[Document, ...], predictions: tuple[Prediction, ...], seed: int = 42
) -> dict:
    """Проверяет две фиксированные половины dev; это не новый независимый holdout."""
    hashes = sorted(
        (d.hash for d in gold),
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    first = set(hashes[: len(hashes) // 2])
    result = {}
    for name, chosen in (("half_a", first), ("half_b", set(hashes) - first)):
        if not chosen:
            continue
        result[name] = evaluate_predictions(
            (d for d in gold if d.hash in chosen), (p for p in predictions if p.hash in chosen)
        ).to_mapping()
    return result


def paired_bootstrap(
    gold: tuple[Document, ...],
    reference: tuple[Prediction, ...],
    candidate: tuple[Prediction, ...],
    seed: int = 42,
    samples: int = 2000,
) -> dict:
    """Оценивает парный разброс дельты F1, не исправляя смещение выбора по dev."""
    counts = []
    for document, old, new in zip(gold, reference, candidate, strict=True):
        row = []
        for prediction in (old, new):
            value = evaluate_predictions((document,), (prediction,)).micro.counts
            row.append((value.tp, value.fp, value.fn))
        counts.append(row)
    matrix = np.asarray(counts)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        total = matrix[rng.integers(0, len(matrix), len(matrix))].sum(axis=0)
        denominator = 2 * total[:, 0] + total[:, 1] + total[:, 2]
        scores = np.divide(2 * total[:, 0], denominator, out=np.zeros(2), where=denominator != 0)
        differences.append(scores[1] - scores[0])
    return {
        "samples": samples,
        "seed": seed,
        "unit": "document, paired",
        "delta_ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
        "fraction_positive": float(np.mean(np.asarray(differences) > 0)),
        "selection_bias_corrected": False,
    }
