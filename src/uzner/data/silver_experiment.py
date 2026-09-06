"""Подготовка отдельной исследовательской абляции согласованной silver-разметки."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from uzner.data.annotation_selection_leakage import SimpleLeakage
from uzner.data.curation.leakage import digest, lexical_key
from uzner.data.io import read_jsonl, sha256_file
from uzner.domain import Document
from uzner.experiments.artifacts import write_json, write_jsonl


@dataclass(frozen=True)
class SilverPreparation:
    """Неизменяемые входы и отдельный каталог исследовательской выборки."""

    release: Path
    official_root: Path
    output: Path
    research_authorized: bool = False


@dataclass(frozen=True)
class SilverCounts:
    """Числа принятых и исключённых документов для отчёта."""

    input_rows: int
    accepted: int
    empty: int
    excluded: dict[str, int]
    sources: dict[str, int]


def prepare_silver(request: SilverPreparation) -> SilverCounts:
    """Проверяет релиз, исключает пересечения и сохраняет исходные флаги качества."""
    if not request.research_authorized:
        raise ValueError("Нужно явное разрешение исследовательского использования silver")
    if request.output.exists():
        raise FileExistsError(f"Выходной каталог уже существует: {request.output}")
    source = request.release / "silver_candidates.jsonl"
    summary_path = request.release / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if sha256_file(source) != summary["output_sha256"][source.name]:
        raise ValueError("SHA-256 silver не совпадает с опубликованным релизом")
    rows = read_jsonl(source)
    if len(rows) != summary["silver_candidates"]:
        raise ValueError("Число записей не совпадает с manifest релиза")
    leakage = SimpleLeakage(request.official_root)
    official_ids = {
        row["hash"]
        for split in ("train", "dev")
        for row in read_jsonl(request.official_root / f"{split}.jsonl")
    }
    accepted, rejected = [], []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for row in rows:
        document = Document.from_mapping(row, source="external_silver")
        if (
            row.get("annotation_status") != "two_pass_exact_agreement"
            or row.get("kind") != "pseudo"
            or not isinstance(row.get("source_hash"), str)
            or not row["source_hash"]
            or "entities" not in row
        ):
            raise ValueError(f"Неверный контракт silver: {document.hash}")
        if digest(document.text) != row.get("meta", {}).get("content_sha256"):
            raise ValueError(f"Исходный текст изменён: {document.hash}")
        key = digest(lexical_key(document.text))
        hit = leakage.check(document.text)
        reason = None
        if document.hash in official_ids or row["source_hash"] in official_ids:
            reason = "official_identifier"
        elif hit is not None:
            reason = hit.reason
        elif not document.text.strip():
            reason = "empty_text"
        elif document.hash in seen_ids or key in seen_texts:
            reason = "duplicate"
        if reason:
            rejected.append({"hash": document.hash, "reason": reason})
            continue
        seen_ids.add(document.hash)
        seen_texts.add(key)
        accepted.append(row)
    if not accepted:
        raise ValueError("После проверок silver-выборка пуста")
    counts = SilverCounts(
        input_rows=len(rows),
        accepted=len(accepted),
        empty=sum(not row["entities"] for row in accepted),
        excluded=dict(Counter(row["reason"] for row in rejected)),
        sources=dict(Counter(row["meta"].get("source", "unknown") for row in accepted)),
    )
    destination = request.output / "silver.jsonl"
    write_jsonl(destination, accepted)
    write_jsonl(request.output / "excluded.jsonl", rejected)
    write_json(
        request.output / "manifest.json",
        {
            "counts": asdict(counts),
            "source": str(source),
            "source_sha256": sha256_file(source),
            "summary_sha256": sha256_file(summary_path),
            "output_sha256": sha256_file(destination),
            "official_sha256": {
                split: sha256_file(request.official_root / f"{split}.jsonl")
                for split in ("train", "dev")
            },
            "research_authorized": True,
            "human_validated": False,
            "rights_approved": False,
            "scope": "User-requested research ablation only; original release gates preserved",
            "leakage_check": "SimpleLeakage: normalized exact, 10-word overlap, anchored dev near",
            "caveat": (
                "Same-model agreement is not independent gold accuracy; not a legal clearance"
            ),
        },
    )
    return counts
