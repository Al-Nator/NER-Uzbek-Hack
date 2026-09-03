"""Интеграционные проверки зафиксированных данных и конфигов."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from uzner.config import load_data_config, load_experiment_config, resolve_sources
from uzner.data.io import load_documents, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = PROJECT_ROOT / "configs/data/official.yaml"
MANIFEST = PROJECT_ROOT / "ner_uz_hackathon_participant/data/dataset_manifest.json"


def test_official_data_matches_organizer_manifest() -> None:
    """Проверяет hashes, размеры и разметку official split-ов."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data = load_data_config(DATA_CONFIG)

    for split in ("train", "dev"):
        sources = resolve_sources(data, split=split, project_root=PROJECT_ROOT)
        assert len(sources) == 1
        _, path = sources[0]
        expected = manifest["splits"][split]
        documents = load_documents(sources)
        labels = Counter(entity.label for document in documents for entity in document.entities)

        assert sha256_file(path) == expected["sha256"]
        assert len(documents) == expected["records"]
        assert (
            sum(bool(document.entities) for document in documents)
            == expected["records_with_entities"]
        )
        assert sum(labels.values()) == expected["entities"]
        assert dict(labels) == expected["entities_by_label"]


def test_all_experiment_configs_reference_existing_inputs() -> None:
    """Проверяет все зафиксированные experiment YAML-файлы."""
    config_paths = sorted((PROJECT_ROOT / "configs/experiments").glob("*.yaml"))

    assert config_paths
    for config_path in config_paths:
        experiment = load_experiment_config(config_path)
        assert experiment.run_id == config_path.stem
        assert len(experiment.encoder.revision) == 40
        assert experiment.training.batch_size * experiment.training.gradient_accumulation_steps == 8
        data_path = PROJECT_ROOT / experiment.data_config
        assert data_path.is_file()
        data = load_data_config(data_path)
        for split in ("train", "dev"):
            sources = resolve_sources(data, split=split, project_root=PROJECT_ROOT)
            assert sources
            assert all(path.is_file() for _, path in sources)
