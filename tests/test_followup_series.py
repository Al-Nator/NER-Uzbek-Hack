"""Контракты полного релиза и однофакторных новых экспериментов."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from uzner.config import TrainingConfig, load_experiment_config
from uzner.data.augmentation import METHODS, load_candidates
from uzner.data.augmentation_full import prepare_full
from uzner.domain import Document, Entity
from uzner.experiments.artifacts import write_jsonl
from uzner.experiments.mlflow_tracking import MlflowTracker
from uzner.experiments.series import load_series
from uzner.training.optimizer import make_optimizer


def row(method, key="a", target="Али", status="ok"):
    """Создаёт структурно правильную копию трёхсимвольной сущности."""
    entity = {"start": 0, "end": 3, "label": "NAME"}
    return {
        "hash": key + method,
        "source_hash": key,
        "variant": method,
        "direction": "latn_to_cyrl",
        "source_text": "Ali",
        "text": target,
        "target_text": target,
        "entities": [entity],
        "source_entities": [entity],
        "target_entities": [{**entity, "surface": target}],
        "status": status,
        "alignments": [{"source_start": 0, "source_end": 3, "target_start": 0, "target_end": 3}],
    }


def test_independent_full_release(tmp_path):
    """Review одной ветки не удаляет допустимую копию другой; dev не меняется."""
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    write_jsonl(
        train,
        [
            {"hash": k, "text": "Ali", "entities": row(METHODS[0])["entities"]}
            for k in ("a", "b", "c", "d")
        ],
    )
    write_jsonl(dev, [{"hash": "dev", "text": "Дев", "entities": []}])
    old = train.read_bytes(), dev.read_bytes()
    for method in METHODS:
        write_jsonl(
            tmp_path / f"{method}.jsonl",
            [
                row(method),
                row(method, "b", "Боб", "review" if method == METHODS[1] else "ok"),
                row(method, "c", "Ali"),
                row(method, "d", "Дев"),
            ],
        )
    out = tmp_path / "release"
    result = prepare_full(tmp_path, train, dev, out)
    assert result["outputs"]["converter_full"]["rows"] == 2
    assert result["outputs"]["luna_full"]["rows"] == 1
    assert result["outputs"]["converter_full"]["train_total"] == 6
    assert result["language_quality_verified"] is False
    assert old == (train.read_bytes(), dev.read_bytes())
    with pytest.raises(FileExistsError):
        prepare_full(tmp_path, train, dev, out)


@pytest.mark.parametrize("damage", ["duplicate", "direction", "source", "offset"])
def test_combined_invalid_inputs_rejected(tmp_path, damage):
    """Полный JSONL не обходит проверку групп, направлений и Unicode-границ."""
    first = row(METHODS[0])
    rows = [first]
    if damage == "duplicate":
        rows.append(first)
    elif damage == "direction":
        first["direction"] = "unknown"
    elif damage == "source":
        first["source_hash"] = "dev"
    else:
        first["alignments"][0]["target_end"] = 2
    write_jsonl(tmp_path / f"{METHODS[0]}.jsonl", rows)
    with pytest.raises(ValueError):
        load_candidates(
            tmp_path,
            {"a": Document("a", "Ali", (Entity(0, 3, "NAME"),))},
            METHODS[0],
            combined=True,
        )


def test_followup_single_factors():
    """Данные, encoder и LR головы меняются отдельно от остальных факторов."""
    base = load_experiment_config(
        Path("configs/experiments/a100/s32_bge_m3_retromae_global_pointer.yaml")
    )
    configs = [
        load_experiment_config(Path(p))
        for p in load_series(Path("configs/series/followup_a100.yaml")).select("all")
    ]
    assert len({c.run_id for c in configs}) == 4
    for c in configs[:2]:
        assert c.training == base.training
        assert c.model == base.model and c.encoder == base.encoder
    assert replace(configs[2], run_id=base.run_id, encoder=base.encoder) == base
    assert replace(configs[3], run_id=base.run_id, training=base.training) == base
    assert configs[3].training == replace(base.training, head_learning_rate=1e-4)


def test_optimizer_groups_and_schedule():
    """Раздельный LR не теряет параметры, WD сохраняется, обе группы затухают вместе."""
    model = nn.ModuleDict({"encoder": nn.Linear(2, 2), "classifier": nn.Linear(2, 1)})
    default = make_optimizer(model, TrainingConfig())
    assert len(default.param_groups) == 1
    optimizer = make_optimizer(model, TrainingConfig(head_learning_rate=1e-4))
    groups = optimizer.param_groups
    assert [g["lr"] for g in groups] == [2e-5, 1e-4]
    params = [id(p) for g in groups for p in g["params"]]
    assert len(params) == len(set(params)) == len(list(model.parameters()))
    assert all(g["weight_decay"] == 0.01 for g in groups)
    schedule = get_linear_schedule_with_warmup(optimizer, 0, 10)
    optimizer.step()
    schedule.step()
    assert schedule.get_last_lr()[1] / schedule.get_last_lr()[0] == pytest.approx(5)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    with pytest.raises(ValueError):
        make_optimizer(nn.Linear(2, 2), TrainingConfig(head_learning_rate=1e-4))


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_head_lr(value):
    """Отбрасывает неконечную и неположительную скорость обучения головы."""
    with pytest.raises(ValueError):
        TrainingConfig(head_learning_rate=value)


def test_both_learning_rates_reach_mlflow():
    """Голова и encoder имеют отдельные графики, а не скрытый LR второй группы."""
    from types import SimpleNamespace

    logged = []
    api = SimpleNamespace(log_metrics=lambda values, step: logged.append((values, step)))
    MlflowTracker(api, "test").log_train_step(
        {"learning_rate": 2e-5, "head_learning_rate": 1e-4}, step=25
    )
    assert logged == [({"optimizer/learning_rate": 2e-5, "optimizer/head_learning_rate": 1e-4}, 25)]
