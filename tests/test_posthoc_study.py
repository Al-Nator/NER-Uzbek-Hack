"""Изолированные проверки файловой серии; MLflow в тестах подменён заглушкой."""

import json
from pathlib import Path

import pytest
import yaml

from uzner.data.io import sha256_file, write_predictions
from uzner.domain import Document, Entity, Prediction
from uzner.evaluation.metrics import evaluate_predictions
from uzner.experiments.artifacts import write_json, write_jsonl
from uzner.posthoc.study import Study, load_inputs, read_source, run_study


class DummyRun:
    """Проверяет артефакты без подключения к настоящему MLflow."""

    def __init__(self, root, config, inputs, kind):
        """Запоминает входы обычного протокола research run."""
        self.root, self.config, self.inputs = root, config, inputs
        self.run_id = "test-only-not-in-mlflow"

    def __enter__(self):
        """Создаёт минимальный проверяемый журнал в pytest tmp_path."""
        if self.root.exists():
            raise FileExistsError(self.root)
        write_json(self.root / "resolved_config.json", self.config)
        write_json(
            self.root / "metadata.json", {"inputs": {str(p): sha256_file(p) for p in self.inputs}}
        )
        return self

    def evaluate(self, gold, predictions, train, prefix):
        """Использует реальный evaluator, но не отправляет тестовые метрики наружу."""
        write_predictions(self.root / "predictions/dev.jsonl", predictions)
        return evaluate_predictions(gold, predictions).micro.f1

    def log(self, values):
        """Не публикует unit-тесты в MLflow."""

    def __exit__(self, kind, error, traceback):
        """Завершает тестовый файловый run."""
        write_json(self.root / "status.json", {"status": "complete" if kind is None else "failed"})


@pytest.fixture
def study_config(tmp_path, monkeypatch):
    """Создаёт локальный toy-набор, который невозможно спутать с экспериментом."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("uzner.posthoc.study.ResearchRun", DummyRun)
    train, dev, source, candidates = (
        Path(n) for n in ("train.jsonl", "dev.jsonl", "pred.jsonl", "gp.jsonl")
    )
    write_jsonl(
        train, [{"hash": "t", "text": "Ali", "entities": [{"start": 0, "end": 3, "label": "NAME"}]}]
    )
    write_jsonl(dev, [{"hash": "d1", "text": "Ali"}, {"hash": "d2", "text": "X"}])
    write_predictions(source, (Prediction("d2"), Prediction("d1", (Entity(0, 3, "NAME"),))))
    write_jsonl(
        candidates,
        [
            {"hash": "d1", "entities": [{"start": 0, "end": 3, "label": "NAME", "score": 0.9}]},
            {"hash": "d2", "entities": []},
        ],
    )
    write_json(
        candidates.with_suffix(".manifest.json"),
        {"input_sha256": sha256_file(dev), "output_sha256": sha256_file(candidates), "floor": 0.05},
    )
    payload = {
        "name": "test_posthoc",
        "train": str(train),
        "dev": str(dev),
        "candidates": str(candidates),
        "sources": {n: str(source) for n in ("s33", "s21", "s31", "s62", "s45", "s48")},
        "variants": [{"name": "control"}, {"name": "repeat", "operation": "repeat"}],
    }
    path = Path("study.yaml")
    path.write_text(yaml.safe_dump(payload), "utf-8")
    return path


def test_study_end_to_end_and_resume(study_config):
    """Серия сохраняет артефакты, проверяет provenance и не перезаписывает результат."""
    report = run_study(study_config)
    records = json.loads(report.read_text())["runs"]
    assert len(records) == 2
    assert all(r["mlflow_run_id"] == "test-only-not-in-mlflow" for r in records)
    assert report == run_study(study_config, skip_complete=True)
    with pytest.raises(FileExistsError):
        run_study(study_config)
    with Path("train.jsonl").open("a") as stream:
        stream.write('{"hash":"more","text":"X"}\n')
    with pytest.raises(ValueError, match="Входы"):
        run_study(study_config, skip_complete=True)


def test_cache_manifest_mismatch(study_config):
    """Подмена scored cache обнаруживается по SHA256 до декодирования."""
    study = Study.load(study_config)
    with study.candidates.open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="манифестом"):
        load_inputs(study)


def test_duplicate_variant_and_missing_hash(study_config):
    """Дубликаты run name и неполное покрытие документов запрещены."""
    payload = yaml.safe_load(study_config.read_text())
    payload["variants"] *= 2
    study_config.write_text(yaml.safe_dump(payload), "utf-8")
    with pytest.raises(ValueError, match="Повторное"):
        Study.load(study_config)
    with pytest.raises(ValueError, match="покрытие"):
        read_source(Path("pred.jsonl"), (Document("unknown", "x"),))
