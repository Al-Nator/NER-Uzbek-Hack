"""Загрузка данных и детерминированные DataLoader-ы."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from uzner.config import ExperimentConfig, load_data_config, resolve_sources
from uzner.data.io import load_documents, sha256_file
from uzner.data.spans import SpanCollator, SpanFeature
from uzner.data.windows import (
    WindowBatch,
    WindowCollator,
    WindowDataset,
    WindowFeature,
)
from uzner.domain import Document


def load_split(
    config: ExperimentConfig,
    project_root: Path,
    split: Literal["train", "dev"],
    limit: int | None,
) -> tuple[Document, ...]:
    """Загружает split и применяет только явный smoke-limit."""
    data = load_data_config((project_root / config.data_config).resolve())
    sources = resolve_sources(data, split=split, project_root=project_root)
    documents = tuple(load_documents(sources, require_entities=True))
    if not documents:
        raise ValueError(f"Split {split} не должен быть пустым")
    if limit is not None:
        if limit < 1:
            raise ValueError("Document limit должен быть положительным")
        documents = documents[:limit]
    return documents


def validate_split_separation(train: tuple[Document, ...], dev: tuple[Document, ...]) -> None:
    """Запрещает одинаковые hash в train и оценочном dev обычного эксперимента."""
    overlap = {doc.hash for doc in train} & {doc.hash for doc in dev}
    if overlap:
        raise ValueError(f"Train/dev пересекаются по hash: {sorted(overlap)[:5]}")


def make_loader(
    features: tuple[WindowFeature, ...],
    tokenizer: Any,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader[WindowBatch]:
    """Создаёт детерминированный DataLoader окон."""
    if not features:
        raise ValueError("Не создано ни одного окна для DataLoader")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer не содержит pad_token_id")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        WindowDataset(features),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=(
            SpanCollator(tokenizer.pad_token_id)
            if isinstance(features[0], SpanFeature)
            else WindowCollator(tokenizer.pad_token_id)
        ),
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def collect_data_hashes(
    config: ExperimentConfig,
    project_root: Path,
) -> dict[str, object]:
    """Собирает пути и SHA-256 всех включённых источников."""
    data = load_data_config((project_root / config.data_config).resolve())
    result: dict[str, object] = {}
    for split in ("train", "dev"):
        for name, path in resolve_sources(data, split=split, project_root=project_root):
            result[name] = {
                "split": split,
                "path": str(path),
                "sha256": sha256_file(path),
            }
    return result
