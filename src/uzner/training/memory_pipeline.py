"""Построение памяти только из train и честная оценка frozen GP+kNN."""

import random
import time
from pathlib import Path

import torch
from torch.nn import functional

from uzner.domain import Document, Prediction
from uzner.evaluation.slices import normalize_surface
from uzner.experiments.artifacts import write_json
from uzner.models.memory_knn import KnnConfig, TokenMemory, interpolate_spans
from uzner.training.frozen import FrozenPredictor
from uzner.training.sidecar_run import SidecarRun
from uzner.training.span_prediction import SpanPredictionAccumulator, SpanWindowScores


def build_memory(
    predictor: FrozenPredictor,
    train: tuple[Document, ...],
    dev_texts: set[str],
    config: KnnConfig,
    run: SidecarRun,
) -> TokenMemory:
    """Равномерно выбирает уникальные токены вместе с O, без изменения priors классов."""
    rng = random.Random(config.seed)
    documents = tuple(d for d in train if normalize_surface(d.text) not in dev_texts)
    write_json(run.root / "train_groups.json", {str(i): d.hash for i, d in enumerate(documents)})
    keys, labels, groups = [], [], []
    seen = set()
    for window in predictor.windows(documents, with_labels=True):
        doc = documents[window.document_index]
        normalized = normalize_surface(doc.text)
        selected = []
        for token, (offset, label) in enumerate(
            zip(window.feature.offsets, window.feature.labels, strict=True)
        ):
            key = (normalized, *offset)
            if label < 0 or offset[0] == offset[1] or key in seen:
                continue
            seen.add(key)
            if rng.random() < config.sampling_rate:
                selected.append(token)
                labels.append(label)
                groups.append(window.document_index)
        if selected:
            keys.append(functional.normalize(window.hidden[selected].float(), dim=-1).half().cpu())
        if window.document_index % 500 == 0 and window.feature.window_index == 0:
            print(
                f"kNN datastore document={window.document_index}/{len(documents)} "
                f"keys={len(labels)}",
                flush=True,
            )
    if not keys:
        raise ValueError("Пустая train-память")
    indices = rng.sample(range(len(labels)), min(len(labels), config.capacity))
    memory = TokenMemory(
        torch.cat(keys)[indices], torch.tensor(labels)[indices], torch.tensor(groups)[indices]
    )
    torch.save(
        {"keys": memory.keys, "labels": memory.labels, "groups": memory.groups},
        run.root / "memory.pt",
    )
    run.log(
        {
            "memory/keys": len(indices),
            "memory/background_fraction": float((memory.labels == 0).float().mean()),
            "memory/excluded_documents": len(train) - len(documents),
            "memory/unique_tokens": len(seen),
        }
    )
    return memory


def predict_memory(
    predictor: FrozenPredictor,
    documents: tuple[Document, ...],
    memory: TokenMemory,
    config: KnnConfig,
) -> tuple[tuple[Prediction, ...], tuple[Prediction, ...]]:
    """Возвращает reference и kNN на одинаковых forward-ах и одинаковом decoder-е."""
    hashes = tuple(d.hash for d in documents)
    base = SpanPredictionAccumulator(0.5, hashes, [], [])
    adjusted = SpanPredictionAccumulator(0.5, hashes, [], [])
    for window in predictor.windows(documents):
        posterior, valid = [], []
        for offset in range(0, len(window.hidden), 128):
            p, v = memory.posterior(window.hidden[offset : offset + 128], -1, config)
            posterior.append(p)
            valid.append(v)
        scores = interpolate_spans(
            window.probabilities, torch.cat(posterior), torch.cat(valid), config.alpha
        )
        base.add(
            window.document_index,
            SpanWindowScores(window.feature.offsets, window.probabilities.cpu()),
        )
        adjusted.add(window.document_index, SpanWindowScores(window.feature.offsets, scores.cpu()))
    return base.finish(), adjusted.finish()


def run_memory(
    source: Path,
    train: tuple[Document, ...],
    dev: tuple[Document, ...],
    config: KnnConfig,
    run: SidecarRun,
) -> None:
    """Исполняет фиксированную абляцию, не подбирая k/alpha/temperature по dev."""
    predictor = FrozenPredictor(source)
    started = time.perf_counter()
    memory = build_memory(predictor, train, {normalize_surface(d.text) for d in dev}, config, run)
    memory = TokenMemory(memory.keys.cuda(), memory.labels.cuda(), memory.groups.cuda())
    run.log({"time/build_seconds": time.perf_counter() - started})
    unlabeled = tuple(Document(d.hash, d.text) for d in dev)
    started = time.perf_counter()
    baseline = predictor.predict(unlabeled)
    baseline_seconds = time.perf_counter() - started
    started = time.perf_counter()
    repeated, predicted = predict_memory(predictor, unlabeled, memory, config)
    if [p.to_mapping() for p in baseline] != [p.to_mapping() for p in repeated]:
        raise ValueError("Нарушена воспроизводимость базового decoder-а")
    run.log(
        {
            "time/baseline_seconds": baseline_seconds,
            "time/knn_seconds": time.perf_counter() - started,
        }
    )
    run.evaluate(dev, baseline, train, prefix="reference")
    run.evaluate(dev, predicted, train)
