"""Проверки неизменности serving-конфигурации, словаря и HTTP-адаптера."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from uzner.domain import Document, Entity, Prediction
from uzner.posthoc.config import Variant
from uzner.posthoc.lexicon import build_lexicon
from uzner.posthoc.rules import dictionary_rule, repeat
from uzner.serving.bundle import load_lexicon, save_lexicon, verify_bundle
from uzner.serving.config import RuntimeConfig
from uzner.serving.measurement import compare_predictions, latency_summary


def test_runtime_config_validation(tmp_path):
    """Отклоняет неподдерживаемые режимы и читает простой конфиг."""
    with pytest.raises(ValueError):
        RuntimeConfig(tmp_path, backend="magic")
    with pytest.raises(ValueError):
        RuntimeConfig(tmp_path, precision="int8")
    with pytest.raises(ValueError):
        RuntimeConfig(tmp_path, window_batch=0)
    with pytest.raises(ValueError):
        RuntimeConfig(tmp_path, backend="tensorrt")
    with pytest.raises(ValueError):
        RuntimeConfig(tmp_path, backend="tensorrt", engine_dir=tmp_path)
    with pytest.raises(ValueError):
        RuntimeConfig(tmp_path, engine_models=("s33",))
    with pytest.raises(ValueError):
        RuntimeConfig(tmp_path, backend="tensorrt", engine_dir=tmp_path, engine_models=("bad",))
    with pytest.raises(ValueError):
        RuntimeConfig(tmp_path, crf_compile=True, crf_cpu=False)
    with pytest.raises(ValueError):
        RuntimeConfig(
            tmp_path, backend="tensorrt", engine_dir=tmp_path, engine_models=("s33", "s33")
        )
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"bundle": str(tmp_path)}))
    assert RuntimeConfig.read(path) == RuntimeConfig(tmp_path)


def test_lexicon_round_trip_preserves_offsets_and_repeats(tmp_path):
    """Сохраняет словарь без изменения исходных Unicode-границ."""
    train = tuple(
        Document(str(i), "İstanbul Oʻzbekiston", (Entity(0, 8, "GEO"), Entity(9, 20, "GEO")))
        for i in range(3)
    )
    lexicon = build_lexicon(train, normalized=True)
    path = tmp_path / "lexicon.json"
    save_lexicon(path, lexicon)
    restored = load_lexicon(path)
    assert asdict(restored) == asdict(lexicon)
    doc = Document("test", "🙂 O’zbekiston O’zbekiston", ())
    variant = Variant("c02", "lex_add", normalized=True, propensity=0.9)
    base = Prediction(doc.hash, ())
    result = dictionary_rule(doc, base, variant, restored)
    result = repeat(doc, result, Variant("exact", "repeat"))
    assert [(e.start, e.end, doc.text[e.start : e.end]) for e in result.entities] == [
        (2, 13, "O’zbekiston"),
        (14, 25, "O’zbekiston"),
    ]


def test_parity_ignores_score_and_order_but_not_spans():
    """Сравнивает множества exact spans и проверяет полноту документов."""
    reference = [Prediction("a", (Entity(0, 1, "NAME", 0.5), Entity(2, 3, "GEO")))]
    actual = [Prediction("a", (Entity(2, 3, "GEO"), Entity(0, 1, "NAME", 0.9)))]
    assert compare_predictions(reference, actual)["changed_documents"] == 0
    changed = [Prediction("a", (Entity(0, 2, "NAME"),))]
    assert compare_predictions(reference, changed)["removed_spans"] == 2
    with pytest.raises(ValueError):
        compare_predictions(reference, [])


def test_latency_nearest_rank_and_invalid_input():
    """Проверяет процентили и отклонение пустой выборки задержек."""
    result = latency_summary([0.01, 0.02, 0.03, 0.04])
    assert result["p50_ms"] == 20
    assert result["p95_ms"] == 40
    with pytest.raises(ValueError):
        latency_summary([])


def test_manifest_rejects_modified_files(tmp_path):
    """Не разрешает использовать изменённый после фиксации ресурс."""
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"x": "bad"}}))
    (tmp_path / "x").write_text("changed")
    with pytest.raises(ValueError):
        verify_bundle(tmp_path)


def test_manifest_rejects_path_outside_bundle(tmp_path):
    """Не разрешает manifest обращаться к файлам вне своего каталога."""
    from uzner.data.io import sha256_file

    outside = tmp_path / "outside.txt"
    outside.write_text("known bytes")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"files": {"../outside.txt": sha256_file(outside)}})
    )
    with pytest.raises(ValueError):
        verify_bundle(bundle)


def test_configuration_freezes_original_three_models():
    """Закрепляет состав ансамбля и выбранный проверенный runtime."""
    from uzner.serving.bundle import SOURCES

    assert [name for name, _ in SOURCES] == ["s33", "s21", "s31"]
    config = RuntimeConfig.read(Path("configs/serving/default.json"))
    assert config.precision == "bf16"
    assert config.window_batch == 16
    assert config.engine_models == ("s33", "s21", "s31")
    assert config.crf_compile
    assert not config.sort_windows


def test_hybrid_profile_keeps_rollback():
    """Сохраняет старый backend без изменения состава и постобработки."""
    primary = RuntimeConfig.read(Path("configs/serving/default.json"))
    rollback = RuntimeConfig.read(Path("configs/serving/hybrid_bf16.json"))
    assert rollback.engine_models == ("s33", "s31")
    assert rollback.bundle == primary.bundle
    assert rollback.precision == primary.precision
    assert rollback.window_batch == primary.window_batch
    assert rollback.crf_compile == primary.crf_compile


@pytest.mark.parametrize("length", [1, 2, 32, 513])
def test_compiled_crf_preserves_path(length):
    """Сравнивает обычный и скомпилированный путь, включая равные scores."""
    import torch

    from uzner.models.crf import LinearChainCrf

    torch.manual_seed(7)
    tags = ("O", "B-NAME", "I-NAME", "E-NAME", "S-NAME")
    crf = LinearChainCrf(tags, "bioes").eval()
    emissions = torch.randn(length, len(tags))
    with torch.inference_mode():
        expected = crf.decode_one(emissions)
        ties = crf.decode_one(torch.zeros_like(emissions))
        crf.compile_cpu_decode()
        assert crf.decode_one(emissions) == expected
        assert crf.decode_one(torch.zeros_like(emissions)) == ties
