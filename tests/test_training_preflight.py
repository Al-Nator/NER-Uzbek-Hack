"""Проверка capacity-probe без большой модели и реальной CUDA."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import preflight_training
from tests.helpers import TinyEncoder
from uzner.config import load_experiment_config
from uzner.models.token_tagger import TokenTagger


def test_probe_updates_parameters_and_checks_nonfinite_loss(monkeypatch, tmp_path) -> None:
    """Probe выполняет backward и два AdamW step, отклоняя NaN."""
    config = load_experiment_config(
        Path("configs/experiments/s24_bge_m3_retromae_bioes_constrained.yaml")
    )
    config = replace(config, training=replace(config.training, bf16=False))
    model = TokenTagger(TinyEncoder(), config.model)
    before = model.classifier.weight.detach().clone()

    def tokenize(texts, **kwargs):
        """Имитирует batch заданной полной длины."""
        shape = (len(texts), kwargs["max_length"])
        return {key: torch.ones(shape, dtype=torch.long) for key in ("input_ids", "attention_mask")}

    monkeypatch.setattr(preflight_training, "load_experiment_config", lambda _: config)
    monkeypatch.setattr(preflight_training, "resolve_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(
        preflight_training, "resolve_pretrained_snapshot", lambda _: SimpleNamespace(path=tmp_path)
    )
    monkeypatch.setattr(
        preflight_training.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenize
    )
    monkeypatch.setattr(TokenTagger, "from_pretrained", lambda *a, **k: model)
    for name in (
        "synchronize",
        "reset_peak_memory_stats",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, lambda *_: 0)
    result = preflight_training.probe(tmp_path)
    assert result.optimizer_steps == 2
    assert result.length == 512
    assert not torch.equal(before, model.classifier.weight)
    assert model.encoder.checkpointing_enabled
    with torch.no_grad():
        model.classifier.weight.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="Неконечный"):
        preflight_training.probe(tmp_path)
