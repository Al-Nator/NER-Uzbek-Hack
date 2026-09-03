"""Тесты строгих experiment/data YAML."""

from pathlib import Path

import pytest
import yaml

from uzner.config import (
    DataConfig,
    DataSourceConfig,
    ModelConfig,
    TrainingConfig,
    load_data_config,
    load_experiment_config,
    resolve_sources,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_repository_configs_are_valid_and_pinned() -> None:
    """Каждый подготовленный experiment YAML проходит строгую загрузку."""
    for path in sorted((PROJECT_ROOT / "configs/experiments").glob("*.yaml")):
        config = load_experiment_config(path)
        assert len(config.encoder.revision) == 40
        assert config.primary_metric == "exact_micro_f1"


def test_data_config_resolves_future_sources(tmp_path: Path) -> None:
    """Новый источник подключается конфигом без изменения Python-кода."""
    config = DataConfig(
        sources=(
            DataSourceConfig("gold", "data/gold.jsonl", "train"),
            DataSourceConfig("synthetic", "data/synthetic.jsonl", "train", "synthetic", 0.3),
            DataSourceConfig("dev", "data/dev.jsonl", "dev"),
        )
    )

    sources = resolve_sources(config, split="train", project_root=tmp_path)

    assert [name for name, _ in sources] == ["gold", "synthetic"]


def test_repository_data_config_is_valid() -> None:
    """Официальный data YAML содержит train и dev."""
    config = load_data_config(PROJECT_ROOT / "configs/data/official.yaml")

    assert {source.split for source in config.sources} == {"train", "dev"}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DataSourceConfig("", "x", "train"),
        lambda: DataSourceConfig("x", "x", "other"),
        lambda: DataSourceConfig("x", "x", "train", "other"),
        lambda: DataSourceConfig("x", "x", "train", weight=0),
        lambda: DataConfig(
            (DataSourceConfig("same", "a", "train"), DataSourceConfig("same", "b", "dev"))
        ),
        lambda: DataConfig((DataSourceConfig("train", "a", "train"),)),
        lambda: DataConfig(
            (DataSourceConfig("train", "a", "train"), DataSourceConfig("dev", "b", "dev")),
            max_length=4,
        ),
        lambda: ModelConfig("span", "bio", "softmax", "greedy"),
        lambda: ModelConfig("token_tagging", "bio", "crf", "greedy"),
        lambda: TrainingConfig(epochs=0),
    ],
)
def test_invalid_configurations_are_rejected(factory: object) -> None:
    """Опечатки и несовместимые настройки останавливаются до обучения."""
    with pytest.raises(ValueError):
        factory()


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    """Неизвестный ключ YAML не игнорируется молча."""
    path = tmp_path / "experiment.yaml"
    value = yaml.safe_load(
        (PROJECT_ROOT / "configs/experiments/e10_xlmr_base_bio.yaml").read_text(encoding="utf-8")
    )
    value["experiment"]["typo"] = True
    path.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(ValueError, match="неизвестные"):
        load_experiment_config(path)
