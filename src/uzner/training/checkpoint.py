"""Самодостаточные model и resume-checkpoint-ы."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer

from uzner.config import ExperimentConfig
from uzner.models.factory import model_class
from uzner.models.token_tagger import TokenTagger


@dataclass(frozen=True, slots=True)
class TrainerState:
    """Минимальное состояние для точного resume."""

    epoch: int
    global_step: int
    best_micro_f1: float
    best_epoch: int
    epochs_without_improvement: int


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Загруженные модель, tokenizer и trainer state."""

    model: TokenTagger
    tokenizer: Any
    state: TrainerState


def _head_state(model: TokenTagger) -> dict[str, torch.Tensor]:
    """Извлекает проекцию и CRF без дублирования encoder-а."""
    return {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if not key.startswith("encoder.")
    }


def _replace_directory(temporary: Path, target: Path) -> None:
    """Заменяет checkpoint-каталог с восстанавливаемой резервной копией."""
    backup = target.with_name(target.name + ".backup")
    if backup.exists():
        raise FileExistsError(f"Сначала восстановите или проверьте резервный checkpoint: {backup}")
    if not temporary.is_dir():
        raise FileNotFoundError(temporary)
    if target.exists():
        target.rename(backup)
    try:
        temporary.rename(target)
    except OSError:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def save_checkpoint(
    path: Path,
    model: TokenTagger,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: TrainerState,
    config: ExperimentConfig,
) -> None:
    """Атомарно сохраняет веса, optimizer, scheduler, RNG и tokenizer."""
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.encoder.save_pretrained(temporary / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(temporary / "tokenizer")
    save_file(_head_state(model), temporary / "head.safetensors")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temporary / "training_state.pt",
    )
    (temporary / "trainer_state.json").write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (temporary / "experiment_config.json").write_text(
        json.dumps(config.to_mapping(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _replace_directory(temporary, path)


def load_model_checkpoint(
    path: Path,
    config: ExperimentConfig,
    device: torch.device,
) -> LoadedCheckpoint:
    """Восстанавливает модель и tokenizer без сетевых запросов."""
    if not path.is_dir():
        raise FileNotFoundError(path)
    encoder = AutoModel.from_pretrained(path / "encoder", local_files_only=True)
    model = model_class(config.model)(encoder, config.model)
    missing, unexpected = model.load_state_dict(load_file(path / "head.safetensors"), strict=False)
    missing_head = [name for name in missing if not name.startswith("encoder.")]
    if missing_head or unexpected:
        raise ValueError(
            f"Несовместимый head checkpoint: missing={missing_head}, unexpected={unexpected}"
        )
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    state = TrainerState(**json.loads((path / "trainer_state.json").read_text("utf-8")))
    return LoadedCheckpoint(model.to(device), tokenizer, state)


def restore_training_state(
    path: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> None:
    """Восстанавливает optimizer, scheduler и все RNG после создания модели."""
    payload = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    random.setstate(payload["python_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
