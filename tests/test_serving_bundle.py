"""Безопасная подготовка поставки без копирования optimizer или изменения эталона."""

import json

import pytest

from uzner.serving import bundle


def _sources(root):
    """Создаёт маленькую имитацию всех обязательных ресурсов поставки."""
    train = root / "ner_uz_hackathon_participant/data/train.jsonl"
    train.parent.mkdir(parents=True)
    train.write_text(
        json.dumps(
            {"hash": "train", "text": "Ali", "entities": [{"label": "NAME", "start": 0, "end": 3}]}
        )
        + "\n"
    )
    for _, run_id in bundle.SOURCES:
        checkpoint = root / "runs" / run_id / "checkpoints/best"
        for name in (
            "encoder/config.json",
            "encoder/model.safetensors",
            "head.safetensors",
            "experiment_config.json",
            "trainer_state.json",
            "training_state.pt",
            "tokenizer/tokenizer_config.json",
            "tokenizer/tokenizer.json",
        ):
            path = checkpoint / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
    for name in (
        "runs/posthoc_20260907_combinations_v1_c02_norm_lex_then_repeat/predictions/dev.jsonl",
        "artifacts/submissions/s62_lexicon_repeats_public_v1/predictions.jsonl",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"hash":"test","entities":[]}\n')


@pytest.mark.parametrize("copy_fallback", [False, True])
def test_bundle_preserves_sources_excludes_optimizer_and_verifies_all_bytes(
    tmp_path, monkeypatch, copy_fallback
):
    """Hardlink/copy варианты создают тот же полный manifest без тяжёлого train-state."""
    _sources(tmp_path)
    if copy_fallback:

        def cross_device(*args):
            """Имитирует разные файловые системы для source и bundle."""
            raise OSError("cross-device")

        monkeypatch.setattr(bundle.os, "link", cross_device)
    output = bundle.prepare_bundle(tmp_path, tmp_path / "bundle")
    manifest = bundle.verify_bundle(output)
    assert manifest["train_documents"] == 1
    assert [row["name"] for row in manifest["sources"]] == ["s33", "s21", "s31"]
    assert not list(output.rglob("training_state.pt"))
    assert len(list((tmp_path / "runs").rglob("training_state.pt"))) == 3
    assert manifest["postprocessing"]["repeat_after"]
    with pytest.raises(FileExistsError):
        bundle.prepare_bundle(tmp_path, output)


@pytest.mark.parametrize("missing", ["head.safetensors", "encoder/model.safetensors"])
def test_bundle_missing_checkpoint_fails_before_output(tmp_path, missing):
    """Пустая копия Git не становится успешным manifest без весов модели."""
    _sources(tmp_path)
    source = tmp_path / "runs" / bundle.SOURCES[0][1] / "checkpoints/best" / missing
    source.unlink()
    with pytest.raises(FileNotFoundError):
        bundle.prepare_bundle(tmp_path, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_bundle_requires_real_train_and_references(tmp_path):
    """Не строит пустой словарь и проверяет reference-файлы до копирования весов."""
    _sources(tmp_path)
    reference = tmp_path / "artifacts/submissions/s62_lexicon_repeats_public_v1/predictions.jsonl"
    reference.unlink()
    with pytest.raises(FileNotFoundError):
        bundle.prepare_bundle(tmp_path, tmp_path / "bundle")
    train = tmp_path / "ner_uz_hackathon_participant/data/train.jsonl"
    train.write_text("")
    with pytest.raises(ValueError, match="пустому"):
        bundle.prepare_bundle(tmp_path, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()
