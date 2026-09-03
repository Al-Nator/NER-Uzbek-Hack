from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

ENTITY_LABELS = ("ORG", "NAME", "GEO")
TAGS = (
    "O",
    "B-ORG",
    "I-ORG",
    "B-NAME",
    "I-NAME",
    "B-GEO",
    "I-GEO",
)
TAG_TO_ID = {tag: index for index, tag in enumerate(TAGS)}
DEFAULT_MODEL = "distilbert/distilbert-base-multilingual-cased"
DEFAULT_MAX_LENGTH = 256
DEFAULT_STRIDE = 64

JsonObject = dict[str, Any]
ModelFeature = dict[str, list[int]]
Offsets = list[tuple[int, int]]


def read_records(
    path: Path,
    *,
    require_entities: bool,
    limit: int | None = None,
) -> list[JsonObject]:
    """Читает JSONL и проверяет поля, необходимые baseline."""

    records: list[JsonObject] = []
    seen_hashes: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: empty line")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            record = _validate_record(raw, path, line_number, require_entities)
            record_hash = record["hash"]
            if record_hash in seen_hashes:
                raise ValueError(f"{path}:{line_number}: duplicate hash {record_hash}")
            seen_hashes.add(record_hash)
            records.append(record)
            if limit is not None and len(records) >= limit:
                break

    if not records:
        raise ValueError(f"{path}: no records")
    return records


def _validate_record(
    raw: Any,
    path: Path,
    line_number: int,
    require_entities: bool,
) -> JsonObject:
    """Проверяет одну запись и отбрасывает неиспользуемые поля."""

    if not isinstance(raw, dict):
        raise ValueError(f"{path}:{line_number}: record must be an object")
    record_hash = raw.get("hash")
    text = raw.get("text")
    if not isinstance(record_hash, str) or not record_hash:
        raise ValueError(f"{path}:{line_number}: hash must be a non-empty string")
    if not isinstance(text, str):
        raise ValueError(f"{path}:{line_number}: text must be a string")

    result: JsonObject = {"hash": record_hash, "text": text}
    if require_entities:
        result["entities"] = _validate_entities(
            raw.get("entities"),
            text,
            f"{path}:{line_number}",
        )
    return result


def _validate_entities(raw: Any, text: str, source: str) -> list[JsonObject]:
    """Проверяет классы, координаты, дубли и пересечения gold-сущностей."""

    if not isinstance(raw, list):
        raise ValueError(f"{source}: entities must be an array")

    entities: list[JsonObject] = []
    seen: set[tuple[str, int, int]] = set()
    for index, entity in enumerate(raw):
        if not isinstance(entity, dict):
            raise ValueError(f"{source}/entities[{index}]: entity must be an object")
        label = entity.get("label")
        start = entity.get("start")
        end = entity.get("end")
        if label not in ENTITY_LABELS:
            raise ValueError(f"{source}/entities[{index}]: invalid label {label!r}")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= len(text)
        ):
            raise ValueError(f"{source}/entities[{index}]: invalid offsets")
        key = (label, start, end)
        if key in seen:
            raise ValueError(f"{source}/entities[{index}]: duplicate entity")
        seen.add(key)
        entities.append({"label": label, "start": start, "end": end})

    entities.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    for left, right in zip(entities, entities[1:], strict=False):
        if right["start"] < left["end"]:
            raise ValueError(f"{source}: overlapping entities are not supported")
    return entities


def set_seed(seed: int, *, seed_cuda: bool) -> None:
    """Фиксирует генераторы случайных чисел baseline."""

    random.seed(seed)  # noqa: S311 - воспроизводимость ML, не криптография
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Выбирает CPU или CUDA и проверяет явно запрошенное устройство."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def load_fast_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    """Загружает fast tokenizer, необходимый для координат символов."""

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("baseline requires a fast tokenizer with offset_mapping support")
    return tokenizer


def validate_window(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    stride: int,
) -> None:
    """Проверяет параметры sliding window относительно special tokens."""

    content_length = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if content_length < 1:
        raise ValueError("max-length is too small for tokenizer special tokens")
    if not 0 <= stride < content_length:
        raise ValueError(f"stride must be between 0 and {content_length - 1}")


def tokenize_windows(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    max_length: int,
    stride: int,
) -> list[tuple[ModelFeature, Offsets]]:
    """Разбивает текст на перекрывающиеся окна и сохраняет координаты токенов."""

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
    )
    input_chunks = encoded["input_ids"]
    offset_chunks = encoded["offset_mapping"]
    if input_chunks and isinstance(input_chunks[0], int):
        input_chunks = [input_chunks]
        offset_chunks = [offset_chunks]

    windows: list[tuple[ModelFeature, Offsets]] = []
    for chunk_index, offsets in enumerate(offset_chunks):
        feature: ModelFeature = {}
        for key in ("input_ids", "attention_mask"):
            if key not in encoded:
                continue
            values = encoded[key]
            feature[key] = values[chunk_index] if values and isinstance(values[0], list) else values
        windows.append((feature, [(int(start), int(end)) for start, end in offsets]))
    return windows


def align_labels(offsets: Offsets, entities: list[JsonObject]) -> list[int]:
    """Переводит символьные spans в BIO-метки токенов одного окна."""

    labels: list[int] = []
    entity_index = 0
    for start, end in offsets:
        if start == end:
            labels.append(-100)
            continue
        while entity_index < len(entities) and entities[entity_index]["end"] <= start:
            entity_index += 1
        if entity_index >= len(entities):
            labels.append(TAG_TO_ID["O"])
            continue

        entity = entities[entity_index]
        if end <= entity["start"] or start >= entity["end"]:
            labels.append(TAG_TO_ID["O"])
            continue
        prefix = "B" if start <= entity["start"] < end else "I"
        labels.append(TAG_TO_ID[f"{prefix}-{entity['label']}"])
    return labels


class TokenizedNerDataset(Dataset):
    """Набор токенизированных sliding windows с BIO-метками."""

    def __init__(
        self,
        records: list[JsonObject],
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int,
        stride: int,
        description: str,
    ) -> None:
        """Создаёт окна для всех записей и выравнивает gold spans с токенами."""

        self.features: list[JsonObject] = []
        for record in tqdm(records, desc=description, unit="doc"):
            windows = tokenize_windows(
                tokenizer,
                record["text"],
                max_length=max_length,
                stride=stride,
            )
            for feature, offsets in windows:
                labels = align_labels(offsets, record["entities"])
                if all(label == -100 for label in labels):
                    continue
                self.features.append({**feature, "labels": labels})

        if not self.features:
            raise ValueError(f"{description}: tokenization produced no trainable windows")

    def __len__(self) -> int:
        """Возвращает количество sliding windows."""

        return len(self.features)

    def __getitem__(self, index: int) -> JsonObject:
        """Возвращает одно окно для DataLoader."""

        return self.features[index]


def decode_bio_tokens(tokens: list[tuple[int, int, str]]) -> list[JsonObject]:
    """Собирает символьные spans из упорядоченных BIO-предсказаний токенов."""

    entities: list[JsonObject] = []
    current: JsonObject | None = None

    def flush() -> None:
        """Добавляет накопленную сущность в результат."""

        nonlocal current
        if current is not None:
            entities.append(current)
            current = None

    for start, end, tag in tokens:
        if tag == "O":
            flush()
            continue
        prefix, separator, label = tag.partition("-")
        if separator != "-" or prefix not in {"B", "I"} or label not in ENTITY_LABELS:
            raise ValueError(f"model returned unsupported tag {tag!r}")

        if prefix == "B" or current is None or current["label"] != label:
            flush()
            current = {"label": label, "start": start, "end": end}
        else:
            current["end"] = max(current["end"], end)
    flush()
    return entities
