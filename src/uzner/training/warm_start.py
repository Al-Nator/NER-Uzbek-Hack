"""Новый low-LR run из готовых весов без восстановления старого schedule."""

import json
from pathlib import Path

import torch

from uzner.config import ExperimentConfig
from uzner.data.io import sha256_file
from uzner.training.checkpoint import LoadedCheckpoint, TrainerState, load_model_checkpoint


def load_initial_checkpoint(
    path: Path, config: ExperimentConfig, device: torch.device
) -> LoadedCheckpoint:
    """Проверяет совместимость и сбрасывает счётчики нового experiment run."""
    source = ExperimentConfig.from_mapping(
        json.loads((path / "experiment_config.json").read_text())
    )
    if (source.encoder, source.model, source.tokenization) != (
        config.encoder,
        config.model,
        config.tokenization,
    ):
        raise ValueError("Начальный checkpoint несовместим с encoder/model/tokenization")
    loaded = load_model_checkpoint(path, config, device)
    return LoadedCheckpoint(loaded.model, loaded.tokenizer, TrainerState(0, 0, -1.0, 0, 0))


def initial_checkpoint_metadata(path: Path) -> dict[str, object]:
    """Фиксирует происхождение весов и явный reset optimizer/scheduler."""
    return {
        "path": str(path),
        "optimizer_state": "reset",
        "scheduler_state": "reset",
        "source_state": json.loads((path / "trainer_state.json").read_text()),
        "weights_sha256": {
            str(item.relative_to(path)): sha256_file(item)
            for item in sorted(path.rglob("*.safetensors"))
        },
    }
