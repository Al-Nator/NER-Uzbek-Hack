"""Проверки независимого low-LR run и порогового декодирования."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from tests.test_training_pipeline import _tiny_project
from uzner.config import load_experiment_config
from uzner.training.engine import TrainRequest, train_experiment
from uzner.training.span_prediction import SpanWindowScores, decode_span_windows
from uzner.training.warm_start import load_initial_checkpoint


def test_s33_preserves_recipe_except_continuation_budget() -> None:
    """Конфиг s33 меняет только начальные веса, LR, warmup и число эпох."""
    root = Path("configs/experiments/a100")
    source = load_experiment_config(root / "s32_bge_m3_retromae_global_pointer.yaml")
    target = load_experiment_config(root / "s33_bge_global_pointer_low_lr.yaml")
    assert target.model == source.model
    assert target.encoder == source.encoder
    assert target.tokenization == source.tokenization
    assert target.data_config == source.data_config
    assert target.training == replace(
        source.training,
        epochs=2,
        learning_rate=2e-6,
        warmup_ratio=0.0,
        initial_checkpoint=target.training.initial_checkpoint,
    )
    assert target.training.initial_checkpoint.endswith("/checkpoints/best")


def test_low_lr_starts_new_run_and_preserves_source(tmp_path) -> None:
    """Новый run сохраняет исходник и не наследует нулевой LR или номер эпохи."""
    path = _tiny_project(tmp_path, epochs=1)
    mapping = yaml.safe_load(path.read_text())
    mapping["experiment"]["model"] = {
        "architecture": "span",
        "head": "global_pointer",
        "decoder": "span",
    }
    path.write_text(yaml.safe_dump(mapping))
    first = train_experiment(
        TrainRequest(
            path,
            tmp_path,
            max_epochs=1,
            publish_summary=False,
            output_root_override=tmp_path / "isolated",
        )
    )
    before = (first.paths.best_checkpoint / "trainer_state.json").read_bytes()
    config = load_experiment_config(path)
    loaded = load_initial_checkpoint(first.paths.best_checkpoint, config, torch.device("cpu"))
    assert loaded.state.epoch == loaded.state.global_step == 0
    with pytest.raises(ValueError, match="несовместим"):
        load_initial_checkpoint(
            first.paths.best_checkpoint,
            replace(config, model=replace(config.model, span_threshold=0.4)),
            torch.device("cpu"),
        )
    mapping["experiment"]["run_id"] = "low_lr"
    mapping["experiment"]["training"].update(
        {
            "initial_checkpoint": str(first.paths.best_checkpoint),
            "learning_rate": 2e-6,
            "warmup_ratio": 0.0,
        }
    )
    path.write_text(yaml.safe_dump(mapping))
    second = train_experiment(
        TrainRequest(
            path,
            tmp_path,
            max_epochs=1,
            publish_summary=False,
            output_root_override=tmp_path / "isolated",
        )
    )
    assert before == (first.paths.best_checkpoint / "trainer_state.json").read_bytes()
    assert second.paths.root != first.paths.root
    meta = json.loads(second.paths.metadata.read_text())["initial_checkpoint"]
    assert meta["optimizer_state"] == "reset"
    assert meta["source_state"]["epoch"] == 1
    assert meta["weights_sha256"]


def test_threshold_recovers_candidates_and_averages_negative_votes() -> None:
    """Низкий порог возвращает потерянные кандидаты с учётом всех окон."""
    a = torch.zeros(3, 2, 2)
    b = torch.zeros_like(a)
    a[0, 0, 1], b[0, 0, 1] = 0.8, 0.1
    windows = [SpanWindowScores(((0, 2), (3, 5)), scores) for scores in (a, b)]
    assert not decode_span_windows("doc", windows, 0.5).entities
    low = decode_span_windows("doc", windows, 0.4).entities
    assert len(low) == 1
    assert low[0].score == pytest.approx(0.45)
