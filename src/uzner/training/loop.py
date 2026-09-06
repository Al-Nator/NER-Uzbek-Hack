"""Одна эпоха BF16-обучения с подробными runtime-метриками."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from uzner.config import TrainingConfig
from uzner.data.windows import WindowBatch
from uzner.experiments.logging import RunLogger
from uzner.models.token_tagger import TokenTagger


@dataclass(frozen=True, slots=True)
class TrainEpochResult:
    """Итоги одной эпохи и новый global step."""

    loss: float
    seconds: float
    tokens_per_second: float
    mean_gradient_norm: float
    peak_gpu_memory_gib: float
    global_step: int


def _gpu_memory_gib(device: torch.device) -> float:
    """Возвращает peak allocated CUDA memory в GiB."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**3


def train_epoch(
    model: TokenTagger,
    loader: DataLoader[WindowBatch],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    config: TrainingConfig,
    logger: RunLogger,
    *,
    epoch: int,
    global_step: int,
) -> TrainEpochResult:
    """Обучает модель одну эпоху и логирует optimizer steps."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    interval_started = started
    weighted_loss = 0.0
    token_count = 0
    interval_tokens = 0
    gradient_norms: list[float] = []
    last_loss = 0.0
    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        labels = batch.labels
        if labels is None:
            raise ValueError("Train batch должен содержать labels")
        tokens = int((batch.attention_mask if labels.ndim == 4 else labels >= 0).sum().item())
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=config.bf16 and device.type == "cuda",
        ):
            output = model(
                batch.input_ids,
                batch.attention_mask,
                batch.token_type_ids,
                labels,
            )
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError(f"Неконечный train loss на batch {batch_index}")
        group_start = ((batch_index - 1) // config.gradient_accumulation_steps) * (
            config.gradient_accumulation_steps
        )
        group_size = min(
            config.gradient_accumulation_steps,
            len(loader) - group_start,
        )
        (output.loss / group_size).backward()
        last_loss = float(output.loss.detach().item())
        weighted_loss += last_loss * tokens
        token_count += tokens
        interval_tokens += tokens
        should_step = batch_index % config.gradient_accumulation_steps == 0 or batch_index == len(
            loader
        )
        if not should_step:
            continue
        norm = float(clip_grad_norm_(model.parameters(), config.max_grad_norm).item())
        gradient_norms.append(norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if global_step % config.log_every_steps == 0 or batch_index == len(loader):
            elapsed = perf_counter() - interval_started
            logger.train_step(
                epoch=epoch,
                step=global_step,
                loss=last_loss,
                learning_rate=float(scheduler.get_last_lr()[0]),
                head_learning_rate=(
                    float(scheduler.get_last_lr()[1])
                    if config.head_learning_rate is not None
                    else None
                ),
                gradient_norm=norm,
                tokens_per_second=interval_tokens / elapsed if elapsed else 0.0,
                gpu_memory_gib=_gpu_memory_gib(device),
                loss_components={
                    name: float(value.detach()) for name, value in output.loss_components.items()
                },
            )
            interval_started = perf_counter()
            interval_tokens = 0
    seconds = perf_counter() - started
    if token_count == 0:
        raise RuntimeError("Train epoch не содержит размеченных токенов")
    return TrainEpochResult(
        loss=weighted_loss / token_count,
        seconds=seconds,
        tokens_per_second=token_count / seconds if seconds else 0.0,
        mean_gradient_norm=(sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0),
        peak_gpu_memory_gib=_gpu_memory_gib(device),
        global_step=global_step,
    )
