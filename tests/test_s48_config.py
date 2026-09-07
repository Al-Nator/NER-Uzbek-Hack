"""Контракт чистого расширения данных лучшей одиночной модели."""

from pathlib import Path

from uzner.config import load_data_config, load_experiment_config


def test_s48_preserves_s45_recipe():
    """Меняются только данные и идентификатор, не рецепт обучения s45."""
    root = Path(__file__).resolve().parents[1]
    old = load_experiment_config(root / "configs/experiments/a100/s45_bge_gp_external_silver.yaml")
    new = load_experiment_config(root / "configs/experiments/s48_bge_gp_silver_interim.yaml")
    assert new.encoder == old.encoder
    assert new.model == old.model
    assert new.training == old.training
    assert new.tokenization == old.tokenization
    data = load_data_config(root / new.data_config)
    assert len(data.sources) == 4
    assert [s.split for s in data.sources] == ["train", "train", "train", "dev"]
