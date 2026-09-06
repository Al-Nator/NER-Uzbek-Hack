"""Подготовка изменённых транслитераций и парного сравнения двух методов."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from uzner.data.io import load_documents, sha256_file
from uzner.domain import Document
from uzner.evaluation.slices import normalize_surface
from uzner.experiments.artifacts import write_json, write_jsonl

DIRECTIONS = ("latn_to_cyrl", "cyrl_to_latn")
METHODS = ("corrected_converter", "luna_high")


@dataclass(frozen=True)
class Candidate:
    """Проверенная копия с группой исходного документа и методом."""

    document: Document
    source_hash: str
    direction: str
    method: str
    input_hash: str

    @property
    def key(self) -> tuple[str, str]:
        """Возвращает ключ честного парного сравнения."""
        return self.source_hash, self.direction

    def to_mapping(self) -> dict:
        """Сохраняет NER-контракт и явное происхождение аугментации."""
        return {
            "hash": self.document.hash,
            "text": self.document.text,
            "entities": [entity.to_mapping() for entity in self.document.entities],
            "source_hash": self.source_hash,
            "direction": self.direction,
            "method": self.method,
            "input_hash": self.input_hash,
            "language_approved": False,
            "split": "train",
        }


def validate_candidate(row: dict, original: Document, method: str) -> Candidate:
    """Проверяет исходник, плоские spans и соответствие границ alignment-ам."""
    target = Document.from_mapping(row, source=method)
    if row["source_text"] != original.text or row["target_text"] != target.text:
        raise ValueError("Подмена исходного или целевого текста")
    source_saved = Document.from_mapping(
        {"hash": original.hash, "text": original.text, "entities": row["source_entities"]},
        source="saved_source",
    )
    if source_saved.entities != original.entities or len(target.entities) != len(original.entities):
        raise ValueError("Изменились исходная разметка или число сущностей")
    boundaries = {0: 0}
    source_cursor = target_cursor = 0
    for item in row["alignments"]:
        if (item["source_start"], item["target_start"]) != (source_cursor, target_cursor):
            raise ValueError("Разрыв или перестановка alignment")
        if not (source_cursor < item["source_end"] <= len(original.text)):
            raise ValueError("Некорректный source alignment")
        if not (target_cursor < item["target_end"] <= len(target.text)):
            raise ValueError("Некорректный target alignment")
        source_cursor, target_cursor = item["source_end"], item["target_end"]
        boundaries[source_cursor] = target_cursor
    if (source_cursor, target_cursor) != (len(original.text), len(target.text)):
        raise ValueError("Неполное покрытие alignment")
    for old, new, saved in zip(
        original.entities, target.entities, row["target_entities"], strict=True
    ):
        if (new.label, new.start, new.end) != (
            old.label,
            boundaries.get(old.start),
            boundaries.get(old.end),
        ) or saved["surface"] != target.text[new.start : new.end]:
            raise ValueError("Повреждены границы, класс или поверхность сущности")
    key = f"{method}:{original.hash}:{row['direction']}:{target.text}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    document = Document(f"s4-translit:{digest}", target.text, target.entities, method)
    return Candidate(document, original.hash, row["direction"], method, row["hash"])


def load_candidates(
    derived: Path, originals: dict[str, Document], method: str, *, combined: bool = False
) -> tuple[dict[tuple[str, str], Candidate], Counter]:
    """Читает snapshot, исключает review и неизменённые записи до обучения."""
    result, seen, counts = {}, set(), Counter()
    paths = (
        [(derived / f"{method}.jsonl", None)]
        if combined
        else [(derived / f"{method}.{direction}.jsonl", direction) for direction in DIRECTIONS]
    )
    for path, direction in paths:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                key = row["source_hash"], row["direction"]
                if (
                    key in seen
                    or key[1] not in DIRECTIONS
                    or (direction is not None and key[1] != direction)
                    or row["variant"] != method
                ):
                    raise ValueError("Дубликат ключа или несоответствие variant/direction")
                seen.add(key)
                counts["input"] += 1
                if key[0] not in originals:
                    raise ValueError("Транслитерация не принадлежит official train")
                if row["source_text"] != originals[key[0]].text:
                    raise ValueError("source_text не совпадает с official train")
                if row["status"] != "ok":
                    counts["review_or_failed"] += 1
                    continue
                candidate = validate_candidate(row, originals[key[0]], method)
                if candidate.document.text == originals[key[0]].text:
                    counts["unchanged"] += 1
                    continue
                result[key] = candidate
                counts["changed_ok"] += 1
    return result, counts


def select_unique(
    pools: tuple[dict[tuple[str, str], Candidate], ...], blocked: set[str]
) -> tuple[list[tuple[str, str]], Counter]:
    """Исключает обе стороны пары при дубле хотя бы в одном методе."""
    common = set(pools[0]).intersection(*(set(pool) for pool in pools[1:]))
    seen = [set(blocked) for _ in pools]
    selected, counts = [], Counter()
    for key in sorted(common):
        surfaces = [normalize_surface(pool[key].document.text) for pool in pools]
        if any(surface in used for surface, used in zip(surfaces, seen, strict=True)):
            counts["duplicate_or_original_or_dev"] += 1
            continue
        selected.append(key)
        for surface, used in zip(surfaces, seen, strict=True):
            used.add(surface)
    counts["common_before_dedup"] = len(common)
    counts["selected"] = len(selected)
    return selected, counts


def prepare(derived: Path, train_path: Path, dev_path: Path, output: Path) -> dict:
    """Создаёт неизменяемый релиз full-converter и двух matched-веток."""
    if output.exists():
        raise FileExistsError(output)
    originals = {doc.hash: doc for doc in load_documents((("official", train_path),))}
    dev = load_documents((("dev", dev_path),))
    if set(originals).intersection(doc.hash for doc in dev):
        raise ValueError("Пересечение official train/dev hash")
    blocked = {normalize_surface(doc.text) for doc in (*originals.values(), *dev)}
    converter, converter_counts = load_candidates(derived, originals, METHODS[0])
    luna, luna_counts = load_candidates(derived, originals, METHODS[1])
    full, full_counts = select_unique((converter,), blocked)
    matched, matched_counts = select_unique((converter, luna), blocked)
    inputs = [
        train_path,
        dev_path,
        *(derived / f"{m}.{d}.jsonl" for m in METHODS for d in DIRECTIONS),
    ]
    manifest = {
        "schema_version": 1,
        "train_originals": len(originals),
        "dev_unchanged": len(dev),
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
        "language_quality_verified": False,
        "selection_uses_dev_labels": False,
        "dedup": "normalize_surface(text); originals unchanged; paired exclusion",
        "source_stats": {METHODS[0]: dict(converter_counts), METHODS[1]: dict(luna_counts)},
        "full": dict(full_counts),
        "matched": dict(matched_counts),
        "outputs": {},
    }
    for name, pool, keys in (
        ("converter_full", converter, full),
        ("converter_matched", converter, matched),
        ("luna_matched", luna, matched),
    ):
        path = output / f"{name}.jsonl"
        write_jsonl(path, (pool[key].to_mapping() for key in keys))
        manifest["outputs"][name] = {
            "rows": len(keys),
            "train_total": len(originals) + len(keys),
            "sha256": sha256_file(path),
            "directions": dict(Counter(key[1] for key in keys)),
            "groups": len({key[0] for key in keys}),
        }
    write_json(output / "manifest.json", manifest)
    return manifest
