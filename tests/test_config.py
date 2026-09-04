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
    with_run_suffix,
)
from uzner.experiments.series import SeriesConfig, SeriesStage, load_series

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_repository_configs_are_valid_and_pinned() -> None:
    """Каждый подготовленный experiment YAML проходит строгую загрузку."""
    for path in sorted((PROJECT_ROOT / "configs/experiments").glob("*.yaml")):
        config = load_experiment_config(path)
        assert len(config.encoder.revision) == 40
        expected = "dev_loss" if config.pipeline == "official_reference" else "exact_micro_f1"
        assert config.primary_metric == expected
        if config.run_id == "e12_mmbert_base_bio":
            assert config.encoder.use_safetensors is False


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


def test_first_series_manifest_is_ordered_and_controlled() -> None:
    """Манифест первой серии ссылается на валидные и сравнимые конфиги."""
    series = load_series(PROJECT_ROOT / "configs/series/first.yaml")
    selected = series.select("all")
    configs = [load_experiment_config(PROJECT_ROOT / path) for path in selected]
    encoders = [config for config in configs if config.run_id.startswith("e1")]

    assert [stage.name for stage in series.stages] == [
        "baseline",
        "encoders",
        "decoding",
    ]
    assert len(selected) == 9
    assert all(config.tokenization.max_length == 512 for config in encoders)
    assert all(
        config.training.batch_size * config.training.gradient_accumulation_steps == 8
        for config in encoders
    )
    with pytest.raises(ValueError, match="stage"):
        series.select("unknown")


def test_second_series_manifest_is_ordered_and_gpu_safe() -> None:
    """Вторая серия переносит decoding и безопасно масштабирует XLM-R на 16 ГБ."""
    series = load_series(PROJECT_ROOT / "configs/series/second.yaml")
    selected = series.select("all")
    configs = [load_experiment_config(PROJECT_ROOT / path) for path in selected]
    mdeberta = configs[:2]
    xlmr_large = configs[2:]

    assert [stage.name for stage in series.stages] == ["mdeberta", "xlmr-large"]
    assert [config.run_id for config in configs] == [
        "s20_mdeberta_v3_base_bioes_constrained",
        "s21_mdeberta_v3_base_bioes_crf",
        "s22_xlmr_large_bioes_constrained",
        "s23_xlmr_large_bioes_crf",
    ]
    assert [config.model.decoder for config in mdeberta] == ["constrained", "crf"]
    assert [config.model.decoder for config in xlmr_large] == ["constrained", "crf"]
    assert all(config.model.tag_scheme == "bioes" for config in configs)
    assert all(config.tokenization.max_length == 512 for config in configs)
    assert all(config.tokenization.stride == 128 for config in configs)
    assert all(config.training.require_gpu and config.training.bf16 for config in configs)
    assert all(
        config.training.batch_size * config.training.gradient_accumulation_steps == 8
        for config in configs
    )
    assert all(config.encoder.name == "microsoft/mdeberta-v3-base" for config in mdeberta)
    assert all(config.encoder.name == "FacebookAI/xlm-roberta-large" for config in xlmr_large)
    assert all(config.training.batch_size == 1 for config in xlmr_large)
    assert all(config.training.gradient_checkpointing for config in xlmr_large)


def test_series_rejects_empty_and_duplicate_entries() -> None:
    """Пустые и повторные этапы не создают неоднозначную серию."""
    with pytest.raises(ValueError, match="name"):
        SeriesStage("", ())
    stage = SeriesStage("same", ("a.yaml",))
    with pytest.raises(ValueError, match="повторяться"):
        SeriesConfig("first", (stage, stage))


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


def test_run_suffix_preserves_config_and_rejects_paths() -> None:
    """Повторный run получает новый ID без возможности выйти из runs/."""
    config = load_experiment_config(PROJECT_ROOT / "configs/experiments/e10_xlmr_base_bio.yaml")

    assert with_run_suffix(config, None) is config
    assert with_run_suffix(config, "offsetfix-v1").run_id.endswith("_offsetfix-v1")
    with pytest.raises(ValueError, match="suffix"):
        with_run_suffix(config, "../bad")
