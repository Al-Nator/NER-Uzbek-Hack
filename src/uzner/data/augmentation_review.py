"""Подготовка экспериментальной аугментации из reviewed-кандидатов Sol."""

import json
from collections import Counter
from pathlib import Path

from uzner.data.augmentation import DIRECTIONS, select_unique, validate_candidate
from uzner.data.io import load_documents, sha256_file
from uzner.evaluation.slices import normalize_surface
from uzner.experiments.artifacts import write_json, write_jsonl


def prepare_review(input_path: Path, train_path: Path, dev_path: Path, output: Path) -> dict:
    """Проверяет alignment и происхождение, не объявляя кандидатов gold-разметкой."""
    if output.exists():
        raise FileExistsError(output)
    originals = {doc.hash: doc for doc in load_documents((("official", train_path),))}
    dev = load_documents((("dev", dev_path),))
    if set(originals).intersection(doc.hash for doc in dev):
        raise ValueError("Пересечение official train/dev hash")
    pool, seen, counts = {}, set(), Counter()
    with input_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            counts["input"] += 1
            key = row["source_hash"], row["direction"]
            if key in seen or key[1] not in DIRECTIONS:
                raise ValueError("Дубликат ключа или неизвестное направление")
            seen.add(key)
            if row["variant"] != "luna_sol_medium_review" or key[0] not in originals:
                raise ValueError("Неизвестный вариант или источник вне official train")
            original = originals[key[0]]
            # В reviewed-релизе сохранены исходный текст и полное отображение границ.
            adapted = {
                **row,
                "target_text": row["text"],
                "source_entities": [entity.to_mapping() for entity in original.entities],
                "target_entities": [
                    {**entity, "surface": row["text"][entity["start"] : entity["end"]]}
                    for entity in row["entities"]
                ],
            }
            candidate = validate_candidate(adapted, original, "sol_review")
            if candidate.document.text == original.text:
                counts["unchanged"] += 1
                continue
            pool[key] = candidate
            counts["changed_valid"] += 1
    blocked = {normalize_surface(doc.text) for doc in (*originals.values(), *dev)}
    keys, selected = select_unique((pool,), blocked)
    if not keys:
        raise ValueError("После фильтрации не осталось аугментаций")
    path = output / "sol_review.jsonl"
    write_jsonl(path, (pool[key].to_mapping() for key in keys))
    manifest = {
        "schema_version": 1,
        "train_originals": len(originals),
        "dev_unchanged": len(dev),
        "rows": len(keys),
        "train_total": len(originals) + len(keys),
        "source_stats": dict(counts),
        "selection": dict(selected),
        "directions": dict(Counter(key[1] for key in keys)),
        "language_quality_verified": False,
        "experimental_use_authorized_by_user": True,
        "selection_uses_dev_labels": False,
        "comparison": "same s44 recipe; review and available augmentation volume differ",
        "input_sha256": {str(p): sha256_file(p) for p in (input_path, train_path, dev_path)},
        "output_sha256": sha256_file(path),
    }
    write_json(output / "manifest.json", manifest)
    return manifest
