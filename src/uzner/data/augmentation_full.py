"""Независимые полные train-релизы из объединённых транслитераций."""

from collections import Counter
from pathlib import Path

from uzner.data.augmentation import METHODS, load_candidates, select_unique
from uzner.data.io import load_documents, sha256_file
from uzner.evaluation.slices import normalize_surface
from uzner.experiments.artifacts import write_json, write_jsonl


def prepare_full(derived: Path, train_path: Path, dev_path: Path, output: Path) -> dict:
    """Проверяет обе полные ветки, исключая review, неизменённые тексты и дубли."""
    if output.exists():
        raise FileExistsError(output)
    originals = {doc.hash: doc for doc in load_documents((("official", train_path),))}
    dev = load_documents((("dev", dev_path),))
    if set(originals).intersection(doc.hash for doc in dev):
        raise ValueError("Пересечение official train/dev hash")
    blocked = {normalize_surface(doc.text) for doc in (*originals.values(), *dev)}
    pools, stats = {}, {}
    for method in METHODS:
        pools[method], stats[method] = load_candidates(derived, originals, method, combined=True)
    manifest = {
        "schema_version": 1,
        "train_originals": len(originals),
        "dev_unchanged": len(dev),
        "language_quality_verified": False,
        "selection_uses_dev_labels": False,
        "comparison": "independent full pools; method and volume are confounded",
        "dedup": "normalize_surface(text); exclude original train and dev texts",
        "input_sha256": {
            str(p): sha256_file(p)
            for p in (train_path, dev_path, *(derived / f"{m}.jsonl" for m in METHODS))
        },
        "outputs": {},
    }
    # Отбор независимый: неполная Luna больше не ограничивает полный конвертер.
    selections = {m: select_unique((pools[m],), blocked) for m in METHODS}
    for method, name in zip(METHODS, ("converter_full", "luna_full"), strict=True):
        keys, counts = selections[method]
        path = output / f"{name}.jsonl"
        write_jsonl(path, (pools[method][key].to_mapping() for key in keys))
        manifest["outputs"][name] = {
            "source_stats": dict(stats[method]),
            "selection": dict(counts),
            "rows": len(keys),
            "train_total": len(originals) + len(keys),
            "sha256": sha256_file(path),
            "directions": dict(Counter(k[1] for k in keys)),
            "groups": len({k[0] for k in keys}),
        }
    write_json(output / "manifest.json", manifest)
    return manifest
