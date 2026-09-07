"""Проверки происхождения holdout-данных и изоляции исследовательских журналов."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from uzner.data.io import sha256_file
from uzner.domain import Document
from uzner.experiments.artifacts import write_json
from uzner.experiments.research_run import ResearchRun
from uzner.training.reranker_cache import verify_holdout_source


def test_holdout_source_rejects_contamination(tmp_path, monkeypatch):
    """Неверный train, изменённые данные и полнообученный warm-start запрещены."""
    split = tmp_path / "split"
    split.mkdir()
    hashes = {}
    for name in ("proposer_train", "proposer_valid", "meta_train", "meta_valid"):
        path = split / f"{name}.jsonl"
        path.write_text(json.dumps({"hash": name, "text": name, "entities": []}) + "\n")
        hashes[name] = sha256_file(path)
    write_json(split / "manifest.json", {"output_sha256": hashes})
    source = tmp_path / "source"
    write_json(source / "checkpoints/best/experiment_config.json", {})
    write_json(source / "metadata.json", {"data": {k: {"sha256": v} for k, v in hashes.items()}})
    config = SimpleNamespace(training=SimpleNamespace(initial_checkpoint=None))
    monkeypatch.setattr(
        "uzner.training.reranker_cache.ExperimentConfig.from_mapping", lambda _: config
    )

    def load(config, root, split_name, limit):
        """Возвращает именно внутренние proposer-документы."""
        name = "proposer_train" if split_name == "train" else "proposer_valid"
        return (Document(name, name),)

    monkeypatch.setattr("uzner.training.reranker_cache.load_split", load)
    verify_holdout_source(source, split)
    config.training.initial_checkpoint = "full_train"
    with pytest.raises(ValueError, match="checkpoint запрещён"):
        verify_holdout_source(source, split)
    config.training.initial_checkpoint = None
    monkeypatch.setattr(
        "uzner.training.reranker_cache.load_split", lambda *a: (Document("meta_train", "leak"),)
    )
    with pytest.raises(ValueError, match="другом наборе"):
        verify_holdout_source(source, split)
    (split / "meta_valid.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="Изменён split"):
        verify_holdout_source(source, split)


def test_research_journal_keeps_weights_out_of_mlflow(tmp_path, monkeypatch):
    """Журнал хранит hashes/lock, но публикует только лёгкие артефакты."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uv.lock").write_text("locked")
    mock = MagicMock()
    mock.start_run.return_value.info.run_id = "test-id"
    monkeypatch.setattr("uzner.experiments.research_run.mlflow", mock)
    root = tmp_path / "run"
    with ResearchRun(root, {"variant": "test"}, (tmp_path / "uv.lock",), "span_reranker") as run:
        (root / "checkpoints").mkdir()
        (root / "checkpoints/best.pt").write_bytes(b"weights")
        run.log({"loss": 1.0}, step=1)
    assert json.loads((root / "status.json").read_text())["status"] == "complete"
    assert (root / "environment/uv.lock").read_text() == "locked"
    assert not any(str(c.args[0]).endswith(".pt") for c in mock.log_artifact.call_args_list)
    with pytest.raises(FileExistsError), ResearchRun(root, {}, (), "span_reranker"):
        pass


def test_return_routes_proposers_and_rerankers_separately(tmp_path, monkeypatch):
    """Возврат не смешивает внутреннюю proposer-оценку и результаты reranker."""
    from scripts import return_seventh

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["return_seventh.py", "--once"])
    config = tmp_path / "configs/seventh/rerankers.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("split_root: cache\nvariants:\n  - run_id: s73_test\n")

    def command(args, **kwargs):
        """Имитирует SSH статусы; внешние команды тест не выполняет."""
        kind = "reranker_proposer" if "s7p" in args[-1] else "span_reranker"
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"status": "complete", "kind": kind})
        )

    monkeypatch.setattr(return_seventh.subprocess, "run", command)
    sync = MagicMock()
    monkeypatch.setattr(return_seventh, "sync_remote_run", sync)
    return_seventh.main()
    assert sync.call_count == 4
    assert [c.args[0].destination_experiment for c in sync.call_args_list] == [
        "uzner-seventh-proposers",
        "uzner-seventh-proposers",
        "uzner-seventh-proposers",
        "uzner-seventh-series",
    ]
