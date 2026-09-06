"""Обучение символьной головы на train при полностью замороженном encoder-е."""

import gc
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional

from uzner.domain import Document, Prediction
from uzner.models.char_boundary import CharBoundaryHead, CharConfig
from uzner.training.char_features import (
    BoundaryExample,
    apply_boundaries,
    boundary_batch,
    collect_boundaries,
)
from uzner.training.frozen import FrozenPredictor
from uzner.training.runtime import set_reproducible_seed
from uzner.training.sidecar_run import SidecarRun


@torch.inference_mode()
def refine(
    model: CharBoundaryHead,
    examples: list[BoundaryExample],
    predictions: tuple[Prediction, ...],
    documents: tuple[Document, ...],
    config: CharConfig,
) -> tuple[Prediction, ...]:
    """Применяет char-голову к кешированным gold-free кандидатам."""
    model.eval()
    positions, confidences = [], []
    device = next(model.parameters()).device
    for offset in range(0, len(examples), config.batch_size):
        batch = boundary_batch(examples[offset : offset + config.batch_size], device)
        values, indices = model(*batch[:4]).float().softmax(-1).max(-1)
        positions.extend(indices.tolist())
        confidences.extend(values.tolist())
    return apply_boundaries(predictions, examples, positions, confidences, documents, config)


def run_character(
    source: Path,
    train: tuple[Document, ...],
    dev: tuple[Document, ...],
    config: CharConfig,
    run: SidecarRun,
) -> None:
    """Обучает только char-head, выбирает best по исходному exact-span dev."""
    set_reproducible_seed(config.seed)
    alphabet = {c: i + 2 for i, c in enumerate(sorted({c for d in train for c in d.text}))}
    predictor = FrozenPredictor(source)
    started = time.perf_counter()
    examples = collect_boundaries(predictor, train, alphabet, config, training=True)
    if not examples:
        raise ValueError("Нет train-примеров символьных границ")
    run.log(
        {
            "data/train_boundaries": len(examples),
            "time/train_cache_seconds": time.perf_counter() - started,
        }
    )
    unlabeled = tuple(Document(d.hash, d.text) for d in dev)
    started = time.perf_counter()
    baseline = predictor.predict(unlabeled)
    baseline_seconds = time.perf_counter() - started
    started = time.perf_counter()
    dev_examples = collect_boundaries(
        predictor, unlabeled, alphabet, config, training=False, predictions=baseline
    )
    run.log(
        {
            "data/dev_boundaries": len(dev_examples),
            "time/baseline_seconds": baseline_seconds,
            "time/dev_cache_seconds": time.perf_counter() - started,
        }
    )
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    model = CharBoundaryHead(len(examples[0].hidden), len(alphabet) + 2, config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    rng = random.Random(config.seed)
    run.evaluate(dev, baseline, train, prefix="reference")
    best, step = -1.0, 0
    checkpoint_dir = run.root / "checkpoints"
    checkpoint_dir.mkdir()
    for epoch in range(1, config.epochs + 1):
        order = list(examples)
        rng.shuffle(order)
        model.train()
        for offset in range(0, len(order), config.batch_size):
            batch = boundary_batch(order[offset : offset + config.batch_size], torch.device("cuda"))
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(*batch[:4]), batch[4])
            if not torch.isfinite(loss):
                raise ValueError("Неконечный char loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 25 == 0:
                run.log(
                    {
                        "train/loss": float(loss.detach()),
                        "train/gradient_norm": float(norm),
                        "train/lr": config.learning_rate,
                        "train/epoch": epoch,
                    },
                    step,
                )
        started = time.perf_counter()
        predicted = refine(model, dev_examples, baseline, unlabeled, config)
        run.log({"time/cached_refiner_seconds": time.perf_counter() - started}, step)
        score = run.evaluate(
            dev, predicted, train, prefix=f"epochs/{epoch}", step=epoch, metric_prefix="epoch"
        )
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "alphabet": alphabet,
            "config": asdict(config),
            "epoch": epoch,
            "step": step,
            "random_state": rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "source": str(source.resolve()),
        }
        torch.save(payload, checkpoint_dir / "last.pt")
        if score > best:
            best = score
            torch.save(payload, checkpoint_dir / "best.pt")
            run.evaluate(dev, predicted, train, prefix="dev", step=epoch)
    run.log({"best/exact_micro_f1": best}, config.epochs)
