"""Контракт локального продолжения серии encoder-ов."""

from dataclasses import asdict
from pathlib import Path

from uzner.config import load_experiment_config
from uzner.experiments.series import load_series

ROOT = Path(__file__).resolve().parents[1]


def test_second_b_preserves_s22_recipe() -> None:
    """Два новых encoder-а сохраняют decoder, данные и training budget s22."""
    reference = load_experiment_config(
        ROOT / "configs/experiments/s22_xlmr_large_bioes_constrained.yaml"
    )
    series = load_series(ROOT / "configs/series/second_b.yaml")
    configs = [load_experiment_config(ROOT / path) for path in series.select("all")]
    assert [config.encoder.name for config in configs] == [
        "BAAI/bge-m3-retromae",
        "facebook/xlm-v-base",
    ]
    assert [stage.name for stage in series.stages] == ["bge", "xlm-v"]
    assert len({config.run_id for config in configs}) == 2
    for config in configs:
        assert config.data_config == reference.data_config
        assert config.model == reference.model
        assert config.tokenization == reference.tokenization
        training = asdict(config.training)
        expected = asdict(reference.training)
        training.pop("optimizer")
        expected.pop("optimizer")
        assert training == expected
        expected_optimizer = (
            "adamw_8bit" if config.encoder.name == "facebook/xlm-v-base" else "adamw"
        )
        assert config.training.optimizer == expected_optimizer
        assert config.encoder.use_safetensors is False
        assert config.encoder.trust_remote_code is False
        assert len(config.encoder.revision) == 40


def test_third_a100_preserves_encoder_and_budget() -> None:
    """Span-серия независима от 2B и использует бюджет A100 reference."""
    reference = load_experiment_config(
        ROOT / "configs/experiments/a100/s22_xlmr_large_bioes_constrained.yaml"
    )
    series = load_series(ROOT / "configs/series/third_a100.yaml")
    configs = [load_experiment_config(ROOT / path) for path in series.select("all")]
    assert [item.run_id for item in configs] == [
        "s30_xlmr_large_biaffine",
        "s31_xlmr_large_global_pointer",
    ]
    for config in configs:
        assert config.encoder == reference.encoder
        assert config.data_config == reference.data_config
        assert config.tokenization == reference.tokenization
        assert config.training.batch_size == 8
        assert config.training.gradient_accumulation_steps == 1
        assert config.training.gradient_checkpointing is False
        assert config.training.learning_rate == reference.training.learning_rate
        assert config.model.architecture == "span"
