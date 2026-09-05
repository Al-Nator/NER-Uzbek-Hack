"""Полный CPU train/eval/checkpoint/resume двух span-архитектур."""

import json

import pytest
import yaml

from tests.test_training_pipeline import _tiny_project
from uzner.training.engine import TrainRequest, train_experiment


@pytest.mark.parametrize("head", ["biaffine", "global_pointer"])
def test_span_pipeline_resume_and_no_gold_leakage(tmp_path, head) -> None:
    """Проверяет сохранение head, оптимизатора, продолжение и общий формат результатов."""
    path = _tiny_project(tmp_path, epochs=2)
    config = yaml.safe_load(path.read_text())
    config["experiment"]["model"] = {"architecture": "span", "head": head, "decoder": "span"}
    path.write_text(yaml.safe_dump(config))
    request = TrainRequest(
        path,
        tmp_path,
        max_epochs=1,
        publish_summary=False,
        output_root_override=tmp_path / "isolated",
    )
    first = train_experiment(request)
    predictions = (first.paths.root / "predictions/dev.jsonl").read_text()
    assert (
        json.loads((first.paths.metrics.parent / "span_coverage.json").read_text())["dev"][
            "recall_ceiling"
        ]
        == 1
    )
    # Завершённый resume должен вернуть те же predictions без нового обучения.
    resumed = train_experiment(
        TrainRequest(
            path,
            tmp_path,
            resume=True,
            max_epochs=1,
            publish_summary=False,
            output_root_override=tmp_path / "isolated",
        )
    )
    assert (resumed.paths.root / "predictions/dev.jsonl").read_text() == predictions
    continued = train_experiment(
        TrainRequest(
            path,
            tmp_path,
            resume=True,
            max_epochs=2,
            publish_summary=False,
            output_root_override=tmp_path / "isolated",
        )
    )
    state = json.loads((continued.paths.last_checkpoint / "trainer_state.json").read_text())
    assert state["epoch"] == 2
    assert not (tmp_path / "reports/experiments.csv").exists()
