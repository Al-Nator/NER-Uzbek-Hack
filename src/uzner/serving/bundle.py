"""Подготовка компактного автономного набора весов, словаря и provenance."""

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

from uzner.data.io import load_documents, sha256_file
from uzner.experiments.artifacts import write_json
from uzner.posthoc.lexicon import Lexicon, LexiconEntry, build_lexicon

SOURCES = (
    ("s33", "s33_bge_global_pointer_low_lr_a100-low-lr-v1"),
    ("s21", "s21_mdeberta_v3_base_bioes_crf_s2-sequence-v1"),
    ("s31", "s31_xlmr_large_global_pointer_a100-continuation-v1"),
)


def save_lexicon(path: Path, lexicon: Lexicon) -> None:
    """Сохраняет статистику train без исходных текстов и без pickle."""
    write_json(
        path,
        {
            "normalized": lexicon.normalized,
            "max_tokens": lexicon.max_tokens,
            "entries": [
                {"key": list(key), **asdict(value)}
                for key, value in sorted(lexicon.entries.items())
            ],
        },
    )


def load_lexicon(path: Path) -> Lexicon:
    """Восстанавливает тот же индекс префиксов без пересчёта статистики."""
    data = json.loads(path.read_text("utf-8"))
    entries = {}
    for row in data["entries"]:
        key = tuple(row.pop("key"))
        entries[key] = LexiconEntry(**row)
    prefixes = frozenset(key[:n] for key in entries for n in range(1, len(key) + 1))
    return Lexicon(entries, prefixes, data["normalized"], data["max_tokens"])


def prepare_bundle(root: Path, output: Path) -> Path:
    """Фиксирует s62+c02; не копирует optimizer и не изменяет оригинальные веса."""
    if output.exists():
        raise FileExistsError(output)
    train_path = root / "ner_uz_hackathon_participant/data/train.jsonl"
    train = load_documents((("train", train_path),), require_entities=True)
    if not train:
        raise ValueError("Нельзя строить словарь по пустому train")
    references = {
        "dev": root
        / "runs/posthoc_20260907_combinations_v1_c02_norm_lex_then_repeat"
        / "predictions/dev.jsonl",
        "public": root / "artifacts/submissions/s62_lexicon_repeats_public_v1/predictions.jsonl",
    }
    for source in references.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    for _, run_id in SOURCES:
        source = root / "runs" / run_id / "checkpoints/best"
        for name in (
            "head.safetensors",
            "experiment_config.json",
            "trainer_state.json",
            "encoder/config.json",
            "tokenizer/tokenizer_config.json",
        ):
            if not (source / name).is_file():
                raise FileNotFoundError(f"Неполный исходный checkpoint: {source / name}")
        if not any((source / "encoder").glob("*.safetensors")):
            raise FileNotFoundError(f"Нет encoder weights: {source / 'encoder'}")
    output.mkdir(parents=True)
    for alias, run_id in SOURCES:
        source = root / "runs" / run_id / "checkpoints/best"
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.name == "training_state.pt":
                continue
            target = output / "models" / alias / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)
    save_lexicon(output / "lexicon.json", build_lexicon(train, normalized=True))
    for name, source in references.items():
        target = output / "reference" / f"{name}.jsonl"
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
    manifest = {
        "schema_version": 1,
        "system": "s62 + train-only normalized lexicon + exact repeats (c02)",
        "public_f1_user_reported": 0.8922,
        "dev_f1_archived": 0.9174477988234533,
        "sources": [{"name": name, "run_id": run} for name, run in SOURCES],
        "train_sha256": sha256_file(train_path),
        "train_documents": len(train),
        "postprocessing": {
            "support": 3,
            "purity": 1.0,
            "propensity": 0.9,
            "min_length": 4,
            "normalized": True,
            "repeat_after": True,
        },
        "files": {
            str(p.relative_to(output)): sha256_file(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    write_json(output / "manifest.json", manifest)
    return output


def verify_bundle(root: Path) -> dict:
    """Проверяет все сохранённые байты перед измерением или публикацией."""
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != expected:
            raise ValueError(f"Нарушена целостность bundle: {name}")
    return manifest
