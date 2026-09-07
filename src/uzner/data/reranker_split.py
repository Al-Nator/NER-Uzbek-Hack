"""Групповое train-holdout разбиение для пилота span-reranker."""

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from uzner.data.curation.leakage import lexical_key
from uzner.data.io import read_jsonl, sha256_file
from uzner.experiments.artifacts import write_json, write_jsonl


@dataclass(frozen=True)
class RerankerSplit:
    """Исходные файлы и новый каталог четырёх непересекающихся частей."""

    train: Path
    dev: Path
    output: Path
    seed: int = 42


def grouped_keys(records: list[dict]) -> list[str]:
    """Объединяет provenance, lexical-дубли и близкие тексты по word-shingles."""
    parent = list(range(len(records)))

    def find(index: int) -> int:
        """Находит представителя компоненты с компрессией пути."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        """Объединяет две компоненты в одну группу."""
        parent[find(left)] = find(right)

    owners = {}
    sketches = defaultdict(list)
    shingles = []
    for index, row in enumerate(records):
        text = lexical_key(row["text"])
        for key in (
            "text:" + text,
            "id:" + row["hash"],
            "id:" + row.get("source_hash", row["hash"]),
        ):
            if key in owners:
                union(index, owners[key])
            owners[key] = index
        words = text.split()
        grams = {" ".join(words[i : i + 5]) for i in range(max(0, len(words) - 4))}
        shingles.append(grams)
        if len(grams) < 8:
            continue
        sketch = sorted(grams, key=lambda g: hashlib.sha256(g.encode()).digest())[:16]
        candidates = {j for gram in sketch for j in sketches[gram]}
        for other in candidates:
            previous = shingles[other]
            if len(grams & previous) / len(grams | previous) >= 0.85:
                union(index, other)
        for gram in sketch:
            sketches[gram].append(index)
    groups = defaultdict(list)
    for index, row in enumerate(records):
        groups[find(index)].append(row["hash"])
    labels = {root: min(hashes) for root, hashes in groups.items()}
    return [labels[find(i)] for i in range(len(records))]


def prepare_reranker_split(request: RerankerSplit) -> dict:
    """Исключает группы dev и разделяет original train на proposer/meta части."""
    if request.output.exists():
        raise FileExistsError(request.output)
    train, dev = read_jsonl(request.train), read_jsonl(request.dev)
    if len({r["hash"] for r in train}) != len(train):
        raise ValueError("Повторный train hash")
    groups = grouped_keys(train + dev)
    dev_groups = set(groups[len(train) :])
    partitions = {
        name: [] for name in ("proposer_train", "proposer_valid", "meta_train", "meta_valid")
    }
    assignments = {}
    excluded = []
    for row, group in zip(train, groups[: len(train)], strict=True):
        if group in dev_groups:
            excluded.append(row["hash"])
            continue
        value = int(hashlib.sha256(f"{request.seed}:{group}".encode()).hexdigest()[:8], 16) / 2**32
        name = (
            "proposer_train"
            if value < 0.72
            else "proposer_valid"
            if value < 0.80
            else "meta_train"
            if value < 0.96
            else "meta_valid"
        )
        partitions[name].append(row)
        assignments[row["hash"]] = {"group": group, "split": name}
    if any(not rows for rows in partitions.values()):
        raise ValueError("Пустая часть разбиения; требуется больший набор")
    request.output.mkdir(parents=True)
    for name, rows in partitions.items():
        write_jsonl(request.output / f"{name}.jsonl", rows)
    manifest = {
        "protocol": "grouped train-holdout pilot, not full OOF",
        "seed": request.seed,
        "grouping": "source_hash + lexical exact + bottom16 5-word shingles Jaccard>=.85",
        "grouping_caveat": "heuristic near-duplicates; not semantic/transliteration equivalence",
        "input_sha256": {"train": sha256_file(request.train), "dev": sha256_file(request.dev)},
        "excluded_dev_overlap": excluded,
        "assignments": assignments,
        "counts": {name: len(rows) for name, rows in partitions.items()},
        "output_sha256": {
            name: sha256_file(request.output / f"{name}.jsonl") for name in partitions
        },
    }
    write_json(request.output / "manifest.json", manifest)
    return manifest
