"""Read-only аудит provenance и символьной разметки внешнего UzNER JSONL."""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path

from uzner.data.io import sha256_file


def normalized_text(text: str) -> str:
    """Строит ключ сравнения, сохраняя исходный текст без изменений."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(folded.translate(str.maketrans("’‘ʻʼ`", "'''''")).split())


@dataclass(slots=True)
class ExternalAudit:
    """Хранит наблюдаемые количества без утверждения качества аннотаций."""

    file: str
    sha256: str
    records: int = 0
    synthetic_original: Counter = field(default_factory=Counter)
    synthetic_primary: Counter = field(default_factory=Counter)
    source_original: Counter = field(default_factory=Counter)
    sources: Counter = field(default_factory=Counter)
    labels: Counter = field(default_factory=Counter)
    script: Counter = field(default_factory=Counter)
    promoted_synthetic: int = 0
    invalid_offsets: int = 0
    surface_mismatch: int = 0
    token_tag_length_mismatch: int = 0
    duplicate_ids: int = 0
    duplicate_normalized_text: int = 0
    official_overlap: Counter = field(default_factory=Counter)
    examples: list[dict] = field(default_factory=list)

    def to_mapping(self) -> dict:
        """Сериализует счётчики без изменения их ключей через dataclasses.asdict."""
        return {
            item.name: dict(value) if isinstance(value, Counter) else value
            for item in fields(self)
            for value in (getattr(self, item.name),)
        }


def audit_external(path: Path, official_keys: dict[str, set[str]]) -> ExternalAudit:
    """Проверяет весь JSONL без изменения, импорта или исправления записей."""
    result = ExternalAudit(path.name, sha256_file(path))
    ids: set[str] = set()
    texts: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            text, meta = record["text"], record.get("meta", {})
            original = meta.get("is_synthetic_original", "unknown")
            primary = meta.get("is_synthetic", "unknown")
            result.records += 1
            result.synthetic_original[str(original).lower()] += 1
            result.synthetic_primary[str(primary).lower()] += 1
            result.promoted_synthetic += original is True and primary is False
            result.source_original[str(meta.get("source_tier_original", "unknown"))] += 1
            result.sources[str(meta.get("source", "unknown"))] += 1
            result.script[str(meta.get("script", "unknown"))] += 1
            identifier, key = str(record["id"]), normalized_text(text)
            result.duplicate_ids += identifier in ids
            result.duplicate_normalized_text += key in texts
            ids.add(identifier)
            texts.add(key)
            for split, keys in official_keys.items():
                result.official_overlap[split] += key in keys
            result.token_tag_length_mismatch += len(record.get("tokens", [])) != len(
                record.get("ner_tags", [])
            )
            for entity in record.get("entities", []):
                result.labels[entity["label"]] += 1
                start, end = entity.get("start"), entity.get("end")
                valid = type(start) is int and type(end) is int and 0 <= start < end <= len(text)
                mismatch = valid and text[start:end] != entity.get("text")
                result.invalid_offsets += not valid
                result.surface_mismatch += mismatch
                if (not valid or mismatch) and len(result.examples) < 5:
                    result.examples.append({"id": identifier, "text": text, "entity": entity})
    return result


def audit_directory(directory: Path, official_root: Path) -> dict:
    """Собирает отчёт по внешним split-ам и exact-normalized пересечениям."""
    keys: dict[str, set[str]] = {}
    for split in ("train", "dev"):
        with (official_root / f"{split}.jsonl").open(encoding="utf-8") as stream:
            keys[split] = {normalized_text(json.loads(line)["text"]) for line in stream}
    files = sorted(directory.glob("uzner_*_bioes.jsonl"))
    if not files:
        raise FileNotFoundError(f"Нет UzNER JSONL в {directory}")
    return {
        "schema_version": 1,
        "normalization": "NFKC/casefold/apostrophes/whitespace; full-text equality only",
        "limitations": [
            "Не проверяет смысл меток и соответствие BIOES символьным entities",
            "Не выявляет near-duplicates и предложения внутри длинных документов",
            "Сведения original/primary являются метаданными поставщика",
        ],
        "official_sha256": {split: sha256_file(official_root / f"{split}.jsonl") for split in keys},
        "files": [audit_external(path, keys).to_mapping() for path in files],
    }
