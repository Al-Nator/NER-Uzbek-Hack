"""Поиск и экономная загрузка неизменённого official kit."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

from huggingface_hub import snapshot_download

from uzner.config import EncoderConfig

OFFICIAL_METADATA_PATTERNS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    "*.model",
)


def load_official_modules(project_root: Path) -> tuple[ModuleType, ModuleType]:
    """Импортирует неизменённый baseline относительно явного корня проекта."""
    kit = project_root / "ner_uz_hackathon_participant" / "baseline"
    if not (kit / "train.py").is_file() or not (kit / "predict.py").is_file():
        raise FileNotFoundError(f"Не найден official baseline: {kit}")
    root_string = str(project_root)
    inserted = root_string not in sys.path
    if inserted:
        sys.path.insert(0, root_string)
    try:
        importlib.invalidate_caches()
        predict = importlib.import_module("ner_uz_hackathon_participant.baseline.predict")
        train = importlib.import_module("ner_uz_hackathon_participant.baseline.train")
    finally:
        if inserted:
            sys.path.remove(root_string)
    return predict, train


def download_official_snapshot(encoder: EncoderConfig) -> Path:
    """Скачивает один формат весов и tokenizer вместо всего model repo."""
    safe_patterns = (*OFFICIAL_METADATA_PATTERNS, "model.safetensors")
    snapshot = Path(
        snapshot_download(
            encoder.name,
            revision=encoder.revision,
            allow_patterns=safe_patterns,
        )
    )
    if (snapshot / "model.safetensors").is_file():
        return snapshot
    binary_patterns = (*OFFICIAL_METADATA_PATTERNS, "pytorch_model.bin")
    return Path(
        snapshot_download(
            encoder.name,
            revision=encoder.revision,
            allow_patterns=binary_patterns,
        )
    )
