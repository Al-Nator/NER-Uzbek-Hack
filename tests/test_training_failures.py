"""Тесты защитных веток train loop и experiment dispatcher."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from tests.helpers import TinyEncoder
from tests.test_training_pipeline import _tiny_project
from uzner.config import ModelConfig, TrainingConfig, load_experiment_config
from uzner.data.windows import WindowCollator, WindowDataset, WindowFeature
from uzner.experiments.artifacts import prepare_run_paths
from uzner.experiments.logging import RunLogger
from uzner.models.token_tagger import TaggerOutput, TokenTagger
from uzner.training.engine import TrainRequest, train_experiment
from uzner.training.loop import train_epoch
from uzner.training.runner import run_experiment


@pytest.mark.train
def test_train_epoch_rejects_missing_labels_and_empty_supervision(tmp_path: Path) -> None:
    """Цикл обучения явно отклоняет неразмеченные batch-и."""
    config = TrainingConfig(
        epochs=1,
        batch_size=1,
        eval_batch_size=1,
        bf16=False,
        require_gpu=False,
        log_every_steps=1,
    )
    model = TokenTagger(
        TinyEncoder(),
        ModelConfig("token_tagging", "bio", "softmax", "greedy", 0.0),
    )
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda _step: 1.0)
    paths = prepare_run_paths(
        load_experiment_config(_tiny_project(tmp_path / "project")),
        project_root=tmp_path / "logs",
    )
    logger = RunLogger("test", paths)
    feature = WindowFeature(0, 0, (1,), (1,), None, ((0, 1),), None)
    loader = DataLoader(
        WindowDataset((feature,)),
        batch_size=1,
        collate_fn=WindowCollator(0),
    )
    with pytest.raises(ValueError, match="labels"):
        train_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            torch.device("cpu"),
            config,
            logger,
            epoch=1,
            global_step=0,
        )

    class FiniteEmptyModel(nn.Module):
        """Возвращает конечный loss без supervised tokens."""

        def __init__(self) -> None:
            """Создаёт единственный обучаемый параметр."""
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))

        def forward(self, *args: object) -> TaggerOutput:
            """Считает нулевой дифференцируемый loss."""
            return TaggerOutput(torch.zeros(1, 1, 7), self.weight * 0)

    empty = WindowFeature(0, 0, (1,), (1,), None, ((0, 0),), (-100,))
    empty_loader = DataLoader(WindowDataset((empty,)), batch_size=1, collate_fn=WindowCollator(0))
    empty_model = FiniteEmptyModel()
    empty_optimizer = AdamW(empty_model.parameters())
    with pytest.raises(RuntimeError, match="не содержит"):
        train_epoch(
            empty_model,  # type: ignore[arg-type]
            empty_loader,
            empty_optimizer,
            LambdaLR(empty_optimizer, lambda _step: 1.0),
            torch.device("cpu"),
            config,
            logger,
            epoch=1,
            global_step=0,
        )


def test_request_guards_and_runner_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke не проникает в runs, а runner выбирает pipeline."""
    config_path = _tiny_project(tmp_path)
    with pytest.raises(ValueError, match="output_root_override"):
        train_experiment(TrainRequest(config_path, tmp_path, max_train_documents=1))
    with pytest.raises(ValueError, match="output_root_override"):
        train_experiment(TrainRequest(config_path, tmp_path, max_epochs=1))
    with pytest.raises(ValueError, match="положительным"):
        train_experiment(
            TrainRequest(
                config_path,
                tmp_path,
                max_epochs=0,
                publish_summary=False,
                output_root_override=tmp_path / "zero-epochs",
            )
        )
    with pytest.raises(ValueError, match="публиковать"):
        train_experiment(
            TrainRequest(config_path, tmp_path, output_root_override=tmp_path / "other")
        )

    sentinel = SimpleNamespace(name="custom")
    monkeypatch.setattr("uzner.training.runner.train_experiment", lambda _request: sentinel)
    assert run_experiment(TrainRequest(config_path, tmp_path)) is sentinel

    official_path = Path(__file__).resolve().parents[1] / "configs/experiments/b00_official.yaml"
    official = SimpleNamespace(name="official")
    monkeypatch.setattr(
        "uzner.training.official.run_official_reference",
        lambda _request: official,
    )
    assert run_experiment(TrainRequest(official_path, tmp_path)) is official
