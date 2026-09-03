"""Тесты стандартной структуры run-артефактов."""

import json
from pathlib import Path

import pytest
import yaml

from uzner.config import load_experiment_config
from uzner.experiments.artifacts import (
    prepare_run_paths,
    write_json,
    write_resolved_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_prepare_run_paths_and_write_metadata(tmp_path: Path) -> None:
    """Новый run получает полную предсказуемую раскладку."""
    config = load_experiment_config(PROJECT_ROOT / "configs/experiments/e10_xlmr_base_bio.yaml")

    paths = prepare_run_paths(config, project_root=tmp_path)
    write_resolved_config(paths.resolved_config, config)
    write_json(paths.metadata, {"status": "created"})

    assert paths.best_checkpoint.parent.is_dir()
    saved_config = yaml.safe_load(paths.resolved_config.read_text(encoding="utf-8"))
    assert saved_config["run_id"] == config.run_id
    assert json.loads(paths.metadata.read_text(encoding="utf-8"))["status"] == "created"


def test_nonempty_run_requires_resume(tmp_path: Path) -> None:
    """Существующий run нельзя случайно перезаписать."""
    config = load_experiment_config(PROJECT_ROOT / "configs/experiments/e10_xlmr_base_bio.yaml")
    paths = prepare_run_paths(config, project_root=tmp_path)
    paths.metadata.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_run_paths(config, project_root=tmp_path)
    assert prepare_run_paths(config, project_root=tmp_path, resume=True) == paths
