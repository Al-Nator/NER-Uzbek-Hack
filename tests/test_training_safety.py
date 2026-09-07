"""Защита организаторских запусков и восстановления checkpoint-ов."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.test_official_runner import _official_config, _patch_official_success
from tests.test_training_pipeline import _tiny_project
from uzner.config import load_experiment_config
from uzner.domain import Document
from uzner.training import engine, official
from uzner.training.checkpoint import _replace_directory
from uzner.training.data_setup import load_split, make_loader, validate_split_separation
from uzner.training.requests import TrainRequest


@pytest.mark.parametrize("broken", ["", '{"hash":"x","text":"Ali"}\n'])
def test_bad_training_data_stops_before_model_and_run(tmp_path, monkeypatch, broken):
    """Пустой или неразмеченный train не создаёт ложный run и не загружает GPU."""
    config = _tiny_project(tmp_path)
    (tmp_path / "data/train.jsonl").write_text(broken)
    monkeypatch.setattr(engine, "resolve_pretrained_snapshot", lambda *args: pytest.fail("model"))
    with pytest.raises(ValueError):
        engine.train_experiment(TrainRequest(config, tmp_path))
    assert not (tmp_path / "runs").exists()


def test_split_leakage_and_empty_windows_have_clear_errors(tmp_path):
    """Не оценивает train как dev и не индексирует нулевой элемент пустого списка."""
    doc = Document("x", "Ali", ())
    with pytest.raises(ValueError, match="пересекаются"):
        validate_split_separation((doc,), (doc,))
    validate_split_separation((doc,), (Document("y", "Ali", ()),))
    with pytest.raises(ValueError, match="окна"):
        make_loader(
            (), SimpleNamespace(pad_token_id=0), batch_size=2, shuffle=False, seed=42, num_workers=0
        )
    config = load_experiment_config(_tiny_project(tmp_path))
    with pytest.raises(ValueError, match="положительным"):
        load_split(config, tmp_path, "train", 0)


@pytest.mark.parametrize("pipeline", ["uzner", "official"])
def test_output_override_without_limits_never_logs_mlflow(tmp_path, monkeypatch, pipeline):
    """Сам факт smoke-output исключает MLflow, даже без max-documents/max-epochs."""
    calls = []

    def no_tracker(*args, **kwargs):
        """Фиксирует флаг до любого возможного обращения к MLflow."""
        calls.append(kwargs["enabled"])
        assert not kwargs["enabled"]
        return None

    if pipeline == "official":
        config = _official_config(tmp_path)
        _patch_official_success(monkeypatch, tmp_path)
        monkeypatch.setattr(official, "start_mlflow_run", no_tracker)
        run = official.run_official_reference
    else:
        config = _tiny_project(tmp_path)
        monkeypatch.setattr(engine, "start_mlflow_run", no_tracker)
        run = engine.train_experiment
    run(
        TrainRequest(
            config, tmp_path, publish_summary=False, output_root_override=tmp_path / "smoke"
        )
    )
    assert calls == [False]
    assert not (tmp_path / "mlruns").exists()


def test_official_max_epochs_is_respected_and_resume_does_not_write(tmp_path, monkeypatch):
    """Official wrapper учитывает лимит эпох и отклоняет resume до создания каталогов."""
    config_path = _official_config(tmp_path)
    _patch_official_success(monkeypatch, tmp_path)
    output = tmp_path / "smoke"
    result = official.run_official_reference(
        TrainRequest(
            config_path, tmp_path, max_epochs=3, output_root_override=output, publish_summary=False
        )
    )
    config = yaml.safe_load(result.paths.resolved_config.read_text())
    assert config["training"]["epochs"] == 3
    forbidden = tmp_path / "resume"
    with pytest.raises(ValueError, match="resume"):
        official.run_official_reference(
            TrainRequest(
                config_path,
                tmp_path,
                resume=True,
                output_root_override=forbidden,
                publish_summary=False,
            )
        )
    assert not forbidden.exists()
    with pytest.raises(ValueError, match="положительным"):
        official.run_official_reference(TrainRequest(config_path, tmp_path, max_epochs=0))
    with pytest.raises(ValueError, match="Smoke-limit"):
        official.run_official_reference(TrainRequest(config_path, tmp_path, max_epochs=1))


def _checkpoint(path, value):
    """Создаёт маркер состояния для имитации сбоя файловой системы."""
    path.mkdir()
    (path / "state.json").write_text(json.dumps({"state": value}))


def test_checkpoint_recovers_old_target_on_rename_failure(tmp_path, monkeypatch):
    """Ошибка публикации нового checkpoint возвращает прежний рабочий каталог."""
    target, temporary = tmp_path / "best", tmp_path / "best.tmp"
    _checkpoint(target, "old")
    _checkpoint(temporary, "new")
    original_rename = Path.rename

    def fail_new(path, destination):
        """Имитирует отказ диска в момент публикации временного checkpoint."""
        if path == temporary:
            raise OSError("disk error")
        return original_rename(path, destination)

    monkeypatch.setattr(Path, "rename", fail_new)
    with pytest.raises(OSError, match="disk error"):
        _replace_directory(temporary, target)
    assert json.loads((target / "state.json").read_text())["state"] == "old"
    assert temporary.exists()
    assert not (tmp_path / "best.backup").exists()


def test_checkpoint_never_deletes_unresolved_backup(tmp_path):
    """Резервная копия после аварии не уничтожается при следующем сохранении."""
    backup = tmp_path / "best.backup"
    _checkpoint(backup, "recoverable")
    temporary = tmp_path / "best.tmp"
    _checkpoint(temporary, "new")
    with pytest.raises(FileExistsError, match="резервный"):
        _replace_directory(temporary, tmp_path / "best")
    assert json.loads((backup / "state.json").read_text())["state"] == "recoverable"
    with pytest.raises(FileNotFoundError):
        _replace_directory(tmp_path / "absent", tmp_path / "other")
