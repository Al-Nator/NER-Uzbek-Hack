"""Локальное разрешение закреплённых Hugging Face snapshot-ов."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from uzner.config import EncoderConfig

MODEL_METADATA_PATTERNS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "*.model",
)


@dataclass(frozen=True, slots=True)
class PretrainedSnapshot:
    """Локальный путь и выбранный файл весов encoder-а."""

    path: Path
    weight_file: str


def _weight_candidates(encoder: EncoderConfig) -> tuple[str, ...]:
    """Возвращает допустимые форматы весов в порядке предпочтения."""
    if encoder.use_safetensors is True:
        return ("model.safetensors",)
    if encoder.use_safetensors is False:
        return ("pytorch_model.bin",)
    return ("model.safetensors", "pytorch_model.bin")


def _local_snapshot(encoder: EncoderConfig, weight_file: str) -> Path | None:
    """Ищет полностью готовый pinned snapshot без сетевых запросов."""
    try:
        path = Path(
            snapshot_download(
                encoder.name,
                revision=encoder.revision,
                allow_patterns=(*MODEL_METADATA_PATTERNS, weight_file),
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError:
        return None
    return path if (path / weight_file).is_file() else None


def resolve_pretrained_snapshot(encoder: EncoderConfig) -> PretrainedSnapshot:
    """Использует локальные веса, а при их отсутствии скачивает один формат."""
    direct = Path(encoder.name)
    if direct.is_dir():
        for weight_file in _weight_candidates(encoder):
            if (direct / weight_file).is_file():
                return PretrainedSnapshot(direct.resolve(), weight_file)
        raise FileNotFoundError(f"В локальном encoder-е нет весов: {direct}")

    candidates = _weight_candidates(encoder)
    for weight_file in candidates:
        cached = _local_snapshot(encoder, weight_file)
        if cached is not None:
            return PretrainedSnapshot(cached, weight_file)

    for weight_file in candidates:
        path = Path(
            snapshot_download(
                encoder.name,
                revision=encoder.revision,
                allow_patterns=(*MODEL_METADATA_PATTERNS, weight_file),
            )
        )
        if (path / weight_file).is_file():
            return PretrainedSnapshot(path, weight_file)
    raise FileNotFoundError(f"Не найдены веса pinned encoder-а: {encoder.name}")
