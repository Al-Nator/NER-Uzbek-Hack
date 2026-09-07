"""Защита эталона и непересечение proposer/meta/control."""

import json
from pathlib import Path

import pytest

from uzner.data.io import sha256_file
from uzner.data.reranker_split import RerankerSplit, grouped_keys, prepare_reranker_split
from uzner.experiments.champion import freeze_champion, verify_champion


def test_group_provenance_exact_and_near():
    """Нормализованные, близкие и связанные source_hash тексты остаются вместе."""
    text = " ".join(f"word{i}" for i in range(40))
    rows = [
        {"hash": "a", "text": text},
        {"hash": "b", "text": text.upper()},
        {"hash": "c", "text": text + " tail"},
        {"hash": "d", "text": "другой алфавит", "source_hash": "a"},
        {"hash": "e", "text": "совсем отдельный текст"},
    ]
    keys = grouped_keys(rows)
    assert len(set(keys[:4])) == 1
    assert keys[4] != keys[0]


def test_split_is_stable_and_excludes_control(tmp_path):
    """Train-группы dev исключаются, а meta не пересекается с обучением proposer."""
    rows = [{"hash": f"h{i}", "text": f"уникальный{i}", "entities": []} for i in range(250)]
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train.write_text("".join(json.dumps(r) + "\n" for r in rows))
    dev.write_text(json.dumps({"hash": "dev", "text": "уникальный0"}) + "\n")
    first = prepare_reranker_split(RerankerSplit(train, dev, tmp_path / "first"))
    second = prepare_reranker_split(RerankerSplit(train, dev, tmp_path / "second"))
    assert first == second
    assert "h0" in first["excluded_dev_overlap"]
    assert sum(first["counts"].values()) == 249
    with pytest.raises(FileExistsError):
        prepare_reranker_split(RerankerSplit(train, dev, tmp_path / "first"))


def test_champion_snapshot_is_independent_and_verified(tmp_path: Path):
    """Изменение исходных весов не меняет release; повторная запись запрещена."""
    submission = tmp_path / "submission"
    submission.mkdir()
    predictions = submission / "predictions.jsonl"
    predictions.write_text('{"hash":"a","entities":[]}\n')
    model = tmp_path / "runs/model/checkpoints/best"
    model.mkdir(parents=True)
    weight = model / "head.safetensors"
    weight.write_bytes(b"original")
    (submission / "predictions.manifest.json").write_text(
        json.dumps(
            {
                "output_sha256": sha256_file(predictions),
                "ensemble_run": "s62",
                "sources": [
                    {
                        "run_id": "model",
                        "checkpoint_sha256": {"head.safetensors": sha256_file(weight)},
                    }
                ],
            }
        )
    )
    for name in ("metrics", "predictions"):
        (tmp_path / "runs/s62" / name).mkdir(parents=True)
    (tmp_path / "uv.lock").write_text("lock")
    target = tmp_path / "release"
    try:
        freeze_champion(tmp_path, submission, target)
        weight.write_bytes(b"changed")
        verify_champion(target)
        assert not target.stat().st_mode & 0o222
        with pytest.raises(FileExistsError):
            freeze_champion(tmp_path, submission, target)
        copy = target / "models/model/head.safetensors"
        copy.chmod(0o644)
        copy.write_bytes(b"tamper")
        with pytest.raises(ValueError, match="Изменён файл"):
            verify_champion(target)
    finally:
        for path in [target, *target.rglob("*")]:
            path.chmod(0o755 if path.is_dir() else 0o644)
