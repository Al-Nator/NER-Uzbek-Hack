"""Общие окна и безопасный warm-start финального train+dev обучения."""

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from uzner.config import ExperimentConfig
from uzner.data.spans import SpanFeature, with_span_targets
from uzner.data.windows import WindowFeature, build_window_features
from uzner.domain import Document
from uzner.models.factory import model_class
from uzner.models.pretrained import resolve_pretrained_snapshot
from uzner.training.data_setup import collect_data_hashes
from uzner.training.warm_start import load_initial_checkpoint


def final_features(
    documents: tuple[Document, ...],
    tokenizer: Any,
    config: ExperimentConfig,
) -> tuple[WindowFeature, ...] | tuple[SpanFeature, ...]:
    """Использует общие токенные или span-цели без отдельного декодирования."""
    features = build_window_features(
        documents,
        tokenizer,
        config.tokenization,
        config.model.tag_scheme,
        with_labels=config.model.architecture == "token_tagging",
    )
    return (
        with_span_targets(features, documents) if config.model.architecture == "span" else features
    )


def initialize_final(
    config: ExperimentConfig,
    root: Path,
    device: torch.device,
    *,
    allow_research_initial: bool = False,
) -> tuple[Any, Any]:
    """Инициализирует pretrained или явно разрешённый завершённый checkpoint."""
    if config.training.initial_checkpoint:
        path = root / config.training.initial_checkpoint
        parent = path.parent.parent
        status = json.loads((parent / "status.json").read_text())
        metadata = json.loads((parent / "metadata.json").read_text())
        if status.get("status") != "complete":
            raise ValueError("Warm-start требует завершённый run")
        current_data = collect_data_hashes(config, root)
        if status.get("kind") != "final_fit":
            if not allow_research_initial or "best_micro_f1" not in status:
                raise ValueError("Warm-start требует завершённый final-fit")
            # Перенос A100 меняет пути, но не содержимое и исходные split-ы.
            source_identity = {
                name: (item["sha256"], item["split"])
                for name, item in metadata["data"].items()
            }
            target_identity = {
                name: (item["sha256"], item["split"])
                for name, item in current_data.items()
            }
            if source_identity != target_identity:
                raise ValueError("Research warm-start требует исходные train/dev")
        elif metadata["data"] != current_data:
            raise ValueError("Warm-start final-fit требует тот же train+dev")
        loaded = load_initial_checkpoint(path, config, device)
        return loaded.model, loaded.tokenizer
    snapshot = resolve_pretrained_snapshot(config.encoder)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot.path, use_fast=True, local_files_only=True, fix_mistral_regex=False
    )
    model = (
        model_class(config.model)
        .from_pretrained(config.encoder, config.model, source=snapshot.path)
        .to(device)
    )
    return model, tokenizer


def publish_final_alias(last: Path, best: Path) -> None:
    """Публикует фиксированный последний checkpoint без второго комплекта весов."""
    if best.exists():
        raise FileExistsError(best)
    # Обучение закончено: обе ссылки неизменяемы, resume в final-fit отсутствует.
    shutil.copytree(last, best, copy_function=os.link)
