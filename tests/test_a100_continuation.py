"""Контракт общей последовательной очереди 2B и серии 3 на A100."""

from dataclasses import asdict
from pathlib import Path

from uzner.config import load_experiment_config
from uzner.experiments.series import load_series


def test_a100_queue_order_and_encoder_recipe() -> None:
    """Очередь сохраняет бюджет s22 и исключает локальный 8-битный optimizer."""
    series = load_series(Path("configs/series/continuation_a100.yaml"))
    configs = [load_experiment_config(Path(path)) for path in series.select("all")]
    assert [config.run_id.split("_")[0] for config in configs] == ["s24", "s25", "s30", "s31"]
    assert series.select("all") == (*series.select("encoders"), *series.select("spans"))
    reference = load_experiment_config(
        Path("configs/experiments/a100/s22_xlmr_large_bioes_constrained.yaml")
    )
    for config in configs:
        assert config.data_config == reference.data_config
        assert config.tokenization == reference.tokenization
        actual, expected = asdict(config.training), asdict(reference.training)
        actual.pop("eval_batch_size")
        expected.pop("eval_batch_size")
        assert actual == expected
        assert config.training.eval_batch_size == 8
    for config in configs[:2]:
        local = load_experiment_config(Path(f"configs/experiments/{config.run_id}.yaml"))
        assert config.encoder == local.encoder
        assert config.model == local.model == reference.model
