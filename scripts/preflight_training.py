"""Проверка BF16 backward и AdamW на полном окне без experiment-артефактов."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from uzner.config import load_experiment_config
from uzner.models.factory import model_class
from uzner.models.pretrained import resolve_pretrained_snapshot
from uzner.training.optimizer import make_optimizer
from uzner.training.runtime import resolve_device, set_reproducible_seed


@dataclass(frozen=True)
class CapacityResult:
    """Результат проверки максимального окна, без оценки качества модели."""

    run_id: str
    length: int
    physical_batch: int
    optimizer_steps: int
    seconds: float
    peak_allocated_gib: float
    peak_reserved_gib: float


def probe(config_path: Path) -> CapacityResult:
    """Проверяет два полных optimizer step с реальными размерами конфигурации."""
    config = load_experiment_config(config_path)
    device = resolve_device(True)
    set_reproducible_seed(config.training.seed)
    snapshot = resolve_pretrained_snapshot(config.encoder)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot.path, local_files_only=True, use_fast=True, fix_mistral_regex=False
    )
    model = model_class(config.model).from_pretrained(
        config.encoder, config.model, source=snapshot.path
    )
    model.to(device).train()
    if config.training.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    encoded = tokenizer(
        ["Ali Toshkentdagi tashkilotda ishlaydi. " * config.tokenization.max_length]
        * config.training.batch_size,
        truncation=True,
        padding="max_length",
        max_length=config.tokenization.max_length,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in encoded.items()}
    labels = torch.zeros_like(inputs["input_ids"])
    if config.model.architecture == "span":
        width = labels.shape[1]
        labels = torch.zeros((config.training.batch_size, 3, width, width), device=device)
        labels.masked_fill_(~torch.ones(width, width, device=device, dtype=torch.bool).triu(), -100)
        labels[:, 0, 1, 2] = 1
    optimizer = make_optimizer(model, config.training)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _step in range(2):
        optimizer.zero_grad(set_to_none=True)
        for _micro in range(config.training.gradient_accumulation_steps):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.training.bf16):
                output = model(**inputs, labels=labels)
                loss = output.loss / config.training.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError("Неконечный loss во время capacity probe")
            loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.max_grad_norm, error_if_nonfinite=True
        )
        optimizer.step()
    torch.cuda.synchronize(device)
    return CapacityResult(
        config.run_id,
        config.tokenization.max_length,
        config.training.batch_size,
        2,
        time.perf_counter() - started,
        torch.cuda.max_memory_allocated(device) / 2**30,
        torch.cuda.max_memory_reserved(device) / 2**30,
    )


def main() -> None:
    """Печатает результат GPU-проверки; не создаёт MLflow run или checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    print(json.dumps(asdict(probe(parser.parse_args().config)), ensure_ascii=False))


if __name__ == "__main__":
    main()
