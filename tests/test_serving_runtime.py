"""Проверка orchestration ансамбля на маленьких настоящих tensor-моделях."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.helpers import FakeTokenizer, TinyEncoder
from uzner.config import ModelConfig, load_experiment_config
from uzner.domain import Document, Entity, Prediction
from uzner.models.factory import model_class
from uzner.posthoc.config import Variant
from uzner.posthoc.lexicon import build_lexicon
from uzner.serving import runtime
from uzner.serving.config import RuntimeConfig
from uzner.serving.runtime import Component, ResidentEnsemble


def _ensemble(tmp_path, *, sort=False):
    """Создаёт CPU-объект только для unit-теста общего декодирования."""
    ensemble = ResidentEnsemble.__new__(ResidentEnsemble)
    ensemble.config = RuntimeConfig(tmp_path, precision="fp32", window_batch=2, sort_windows=sort)
    ensemble.device = torch.device("cpu")
    ensemble.timings = {}
    ensemble.variant = Variant("c02", "lex_add", normalized=True, propensity=0.9, repeat_after=True)
    train = tuple(Document(str(i), "Toshkent", (Entity(0, 8, "GEO"),)) for i in range(3))
    ensemble.lexicon = build_lexicon(train, normalized=True)
    return ensemble


def test_constructor_refuses_silent_cpu_fallback(tmp_path, monkeypatch):
    """Production-сервис не запускает тяжёлый ансамбль на CPU по ошибке."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        ResidentEnsemble(RuntimeConfig(tmp_path))


@pytest.mark.parametrize("verify", [True, False])
def test_constructor_verifies_and_loads_all_three_once(tmp_path, monkeypatch, verify):
    """Проверяет bundle и однократную резидентную загрузку закреплённых голосов."""
    checked, loaded = [], []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(runtime, "verify_bundle", checked.append)
    monkeypatch.setattr(runtime, "load_lexicon", lambda path: "lexicon")
    monkeypatch.setattr(ResidentEnsemble, "_load", lambda self, name: loaded.append(name) or name)
    result = ResidentEnsemble(RuntimeConfig(tmp_path), verify=verify)
    assert checked == ([tmp_path] if verify else [])
    assert loaded == ["s33", "s21", "s31"]
    assert result.components == tuple(loaded)
    assert result.variant.repeat_after


@pytest.mark.parametrize("head", ["softmax", "crf", "global_pointer"])
@pytest.mark.parametrize("sort", [False, True])
def test_window_merge_and_decoder_keep_unicode_bounds(tmp_path, head, sort):
    """Исполняет forward/merge/decode разных heads и крайних окон на tiny encoder."""
    ensemble = _ensemble(tmp_path, sort=sort)
    experiment = load_experiment_config(
        Path("configs/experiments/a100/s32_bge_m3_retromae_global_pointer.yaml")
    )
    model_config = ModelConfig(
        architecture="span" if head == "global_pointer" else "token_tagging",
        tag_scheme="bioes",
        head=head,
        decoder="span" if head == "global_pointer" else ("crf" if head == "crf" else "constrained"),
        dropout=0.0,
        span_head_size=8,
    )
    experiment = replace(
        experiment,
        model=model_config,
        tokenization=replace(experiment.tokenization, max_length=8, stride=2),
    )
    model = model_class(model_config)(TinyEncoder(), model_config).eval()
    component = Component("tiny", model, FakeTokenizer(), experiment)
    docs = (Document("long", "🙂 " + "Али Toshkent " * 5, ()), Document("short", "ACME", ()))
    predictions = ensemble._predict_component(component, docs)
    assert [p.hash for p in predictions] == [d.hash for d in docs]
    for doc, prediction in zip(docs, predictions, strict=True):
        Document(doc.hash, doc.text, prediction.entities)
    assert set(ensemble.timings) == {"tiny_tokenize", "tiny_forward_transfer", "tiny_decode"}


def test_majority_dictionary_then_repeats_without_hash_features(tmp_path):
    """Не вызывает модели для пустых текстов; сохраняет порядок и правила c02."""
    ensemble = _ensemble(tmp_path)
    ensemble.components = ("a", "b", "c")
    seen = []

    def predict(component, docs):
        """Два голоса обнаруживают только первое упоминание человека."""
        seen.append((component, tuple(d.hash for d in docs)))
        return tuple(
            Prediction(doc.hash, (Entity(0, 7, "NAME"),) if component != "c" else ())
            for doc in docs
        )

    ensemble._predict_component = predict
    docs = (Document("empty", " ", ()), Document("different-id", "Alisher Toshkent Alisher", ()))
    predictions = ensemble.predict_documents(docs)
    assert predictions[0].entities == ()
    assert [(e.label, e.start, e.end) for e in predictions[1].entities] == [
        ("NAME", 0, 7),
        ("GEO", 8, 16),
        ("NAME", 17, 24),
    ]
    assert all(hashes == ("different-id",) for _, hashes in seen)
    assert ensemble.predict(["", "  "]) == ((), ())
    assert len(seen) == 3
    assert ensemble.predict(["Alisher Toshkent Alisher"])[0] == predictions[1].entities
    with pytest.raises(ValueError, match="gold"):
        ensemble.predict_documents((Document("bad", "Ali", (Entity(0, 3, "NAME"),)),))


@pytest.mark.parametrize("use_engine", [False, True])
def test_load_keeps_head_crf_and_exact_engine_selection(tmp_path, monkeypatch, use_engine):
    """Гибридный backend заменяет только encoder, оставляет общие classifier и CRF."""
    ensemble = _ensemble(tmp_path)
    ensemble.config = RuntimeConfig(
        tmp_path,
        backend="tensorrt" if use_engine else "torch",
        engine_dir=tmp_path if use_engine else None,
        engine_models=("s21",) if use_engine else (),
        crf_compile=True,
    )
    config = load_experiment_config(
        Path("configs/experiments/a100/s32_bge_m3_retromae_global_pointer.yaml")
    )
    path = tmp_path / "models/s21"
    path.mkdir(parents=True)
    (path / "experiment_config.json").write_text(json.dumps(config.to_mapping()))
    calls = []
    crf = SimpleNamespace(
        cpu=lambda: calls.append("crf_cpu"), compile_cpu_decode=lambda: calls.append("compile")
    )
    model = SimpleNamespace(
        encoder=SimpleNamespace(cpu=lambda: calls.append("encoder_cpu")), crf=crf
    )
    model.eval = lambda: model
    model.requires_grad_ = lambda flag: calls.append(("grad", flag))
    monkeypatch.setattr(
        runtime,
        "load_model_checkpoint",
        lambda *args: SimpleNamespace(model=model, tokenizer=FakeTokenizer()),
    )
    monkeypatch.setattr("uzner.serving.tensorrt_runtime.TensorRTEncoder", lambda *args: "engine")
    component = ensemble._load("s21")
    assert component.model is model
    assert calls[:3] == [("grad", False), "crf_cpu", "compile"]
    assert component.engine == ("engine" if use_engine else None)
    assert ("encoder_cpu" in calls) == use_engine


def test_engine_forward_retains_classifier(tmp_path):
    """Прямой TensorRT hidden tensor проходит через прежнюю обученную голову."""
    ensemble = _ensemble(tmp_path)
    component = SimpleNamespace(
        engine=lambda ids, mask: ids.float().unsqueeze(-1),
        model=SimpleNamespace(classifier=lambda hidden: hidden + 2),
    )
    batch = SimpleNamespace(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.ones(1, 2))
    result = ensemble._forward(component, batch)
    assert result.tolist() == [[[3.0], [4.0]]]
