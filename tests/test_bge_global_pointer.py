"""Контракт переноса GlobalPointer на BGE без изменения рецептуры."""

from dataclasses import asdict
from pathlib import Path

from uzner.config import load_experiment_config
from uzner.experiments.series import load_series
from uzner.models.factory import model_class
from uzner.models.span_tagger import SpanTagger


def test_bge_global_pointer_changes_only_encoder() -> None:
    """Новый run сохраняет s31, меняя только encoder и идентификатор."""
    root = Path("configs/experiments/a100")
    reference = load_experiment_config(root / "s31_xlmr_large_global_pointer.yaml")
    bge = load_experiment_config(root / "s24_bge_m3_retromae_bioes_constrained.yaml")
    series = load_series(Path("configs/series/third_bge_a100.yaml"))
    assert series.select("all") == series.select("bge-global-pointer")
    assert len(series.select("all")) == 1
    config = load_experiment_config(Path(series.select("all")[0]))
    assert config.run_id == "s32_bge_m3_retromae_global_pointer"
    assert config.encoder == bge.encoder
    actual, expected = asdict(config), asdict(reference)
    for key in ("run_id", "encoder"):
        actual.pop(key)
        expected.pop(key)
    assert actual == expected
    assert model_class(config.model) is SpanTagger
