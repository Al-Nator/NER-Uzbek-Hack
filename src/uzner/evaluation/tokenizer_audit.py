"""Аудит theoretical exact-span ceiling и фрагментации tokenizer-а."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import ceil
from statistics import mean
from typing import Any

from uzner.config import EncoderConfig, TokenizationConfig
from uzner.data.tagging import is_exactly_representable
from uzner.data.windows import build_window_features, canonicalize_offsets
from uzner.domain import Document, Entity
from uzner.evaluation.slices import detect_script, has_attached_suffix


@dataclass(frozen=True, slots=True)
class AuditSlice:
    """Счётчики выразимости и фрагментации одного разреза."""

    entities: int
    representable: int
    representability: float
    mean_subwords: float
    p95_subwords: float
    mean_chars_per_subword: float


@dataclass(frozen=True, slots=True)
class TokenizerAudit:
    """Полный отчёт о границах и окнах одного tokenizer-а."""

    encoder: str
    revision: str
    max_length: int
    stride: int
    documents: int
    windows: int
    mean_windows_per_document: float
    max_windows_per_document: int
    overall: AuditSlice
    slices: dict[str, dict[str, AuditSlice]]

    def to_mapping(self) -> dict[str, object]:
        """Сериализует tokenizer audit для run-артефакта."""
        return {
            "encoder": self.encoder,
            "revision": self.revision,
            "max_length": self.max_length,
            "stride": self.stride,
            "documents": self.documents,
            "windows": self.windows,
            "mean_windows_per_document": self.mean_windows_per_document,
            "max_windows_per_document": self.max_windows_per_document,
            "overall": asdict(self.overall),
            "slices": {
                group: {name: asdict(value) for name, value in named.items()}
                for group, named in self.slices.items()
            },
        }


def _percentile(values: list[int], percentile: float) -> float:
    """Считает nearest-rank percentile для непустого набора."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def _make_slice(items: list[tuple[bool, int, int]]) -> AuditSlice:
    """Агрегирует один набор сущностей."""
    lengths = [subwords for _, subwords, _ in items]
    representable = sum(flag for flag, _, _ in items)
    total_subwords = sum(lengths)
    return AuditSlice(
        entities=len(items),
        representable=representable,
        representability=representable / len(items) if items else 0.0,
        mean_subwords=mean(lengths) if lengths else 0.0,
        p95_subwords=_percentile(lengths, 0.95),
        mean_chars_per_subword=(
            sum(characters for _, _, characters in items) / total_subwords
            if total_subwords
            else 0.0
        ),
    )


def _surface_group(text: str, entity: Entity) -> dict[str, str]:
    """Возвращает имена разрезов одной gold-сущности."""
    surface = text[entity.start : entity.end]
    return {
        "label": entity.label,
        "script": detect_script(surface),
        "attached_suffix": "yes" if has_attached_suffix(surface) else "no",
        "apostrophe": "yes" if any(mark in surface for mark in "'\u2019\u02bb\u02bc`") else "no",
        "quotes": "yes" if any(mark in surface for mark in '"“”«»„‟') else "no",
    }


def _document_offsets(tokenizer: Any, text: str) -> tuple[tuple[int, int], ...]:
    """Токенизирует полный документ без truncation и special-токенов."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    return canonicalize_offsets(text, encoded["offset_mapping"])


def audit_tokenizer(
    documents: tuple[Document, ...],
    tokenizer: Any,
    encoder: EncoderConfig,
    tokenization: TokenizationConfig,
) -> TokenizerAudit:
    """Измеряет representability, fragmentation и число окон."""
    if not documents:
        raise ValueError("Tokenizer audit требует непустые документы")
    all_items: list[tuple[bool, int, int]] = []
    grouped: dict[str, dict[str, list[tuple[bool, int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for document in documents:
        offsets = _document_offsets(tokenizer, document.text)
        for entity in document.entities:
            token_count = sum(start < entity.end and end > entity.start for start, end in offsets)
            item = (
                is_exactly_representable(offsets, entity),
                token_count,
                entity.end - entity.start,
            )
            all_items.append(item)
            for group, name in _surface_group(document.text, entity).items():
                grouped[group][name].append(item)
    windows = build_window_features(
        documents,
        tokenizer,
        tokenization,
        "bio",
        with_labels=False,
    )
    counts = [0 for _ in documents]
    for window in windows:
        counts[window.document_index] += 1
    return TokenizerAudit(
        encoder=encoder.name,
        revision=encoder.revision,
        max_length=tokenization.max_length,
        stride=tokenization.stride,
        documents=len(documents),
        windows=len(windows),
        mean_windows_per_document=mean(counts),
        max_windows_per_document=max(counts),
        overall=_make_slice(all_items),
        slices={
            group: {name: _make_slice(items) for name, items in sorted(named.items())}
            for group, named in sorted(grouped.items())
        },
    )
