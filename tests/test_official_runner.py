"""Тесты адаптера неизменённого official baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.helpers import FakeTokenizer
from tests.test_training_pipeline import _tiny_project
from uzner.config import EncoderConfig, load_experiment_config
from uzner.training.engine import TrainRequest
from uzner.training.official import (
    _epoch_record,
    _single_source,
    run_official_reference,
)
from uzner.training.official_kit import download_official_snapshot, load_official_modules

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _official_config(root: Path) -> Path:
    """Переводит tiny project в official_reference без сети."""
    path = _tiny_project(root)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiment = value["experiment"]
    experiment["pipeline"] = "official_reference"
    experiment["primary_metric"] = "dev_loss"
    experiment["model"]["decoder"] = "greedy"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def _patch_official_success(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> None:
    """Заменяет тяжёлые official train/predict локальными дублями."""

    def fake_train(arguments: object) -> Path:
        """Создаёт model и official training summary."""
        output = arguments.output_dir
        model = output / "model"
        model.mkdir(parents=True)
        (model / "weights.bin").write_bytes(b"weights")
        summary = {
            "best_dev_loss": 0.25,
            "history": [
                {"epoch": 1, "train_loss": 0.8, "dev_loss": 0.5},
                {"epoch": 2, "train_loss": 0.4, "dev_loss": 0.25},
            ],
        }
        (output / "training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        print("fake official train")
        return model

    def fake_predict(arguments: object) -> Path:
        """Копирует gold entities как идеальное предсказание."""
        records = [
            json.loads(line) for line in arguments.input.read_text(encoding="utf-8").splitlines()
        ]
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            "".join(
                json.dumps(
                    {"hash": record["hash"], "entities": record["entities"]},
                    ensure_ascii=False,
                )
                + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        print("fake official predict")
        return arguments.output

    monkeypatch.setattr(
        "uzner.training.official.download_official_snapshot",
        lambda _encoder: root / "tiny_model",
    )
    monkeypatch.setattr(
        "uzner.training.official.load_official_modules",
        lambda _root: (SimpleNamespace(run=fake_predict), SimpleNamespace(run=fake_train)),
    )
    monkeypatch.setattr(
        "uzner.training.official.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )


@pytest.mark.train
def test_official_reference_gets_standard_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Official output получает exact metrics, audit, логи и manifest."""
    config_path = _official_config(tmp_path)
    _patch_official_success(monkeypatch, tmp_path)
    result = run_official_reference(
        TrainRequest(
            config_path=config_path,
            project_root=tmp_path,
            publish_summary=False,
            output_root_override=tmp_path / "isolated",
        )
    )

    status = json.loads(result.paths.status.read_text(encoding="utf-8"))
    metadata = json.loads(result.paths.metadata.read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert result.best_epoch == 2
    assert result.best_micro_f1 == 1.0
    assert metadata["selection_metric"] == "dev_loss"
    assert metadata["official_source_unchanged"] is True
    assert result.paths.best_checkpoint.is_dir()
    assert (result.paths.events.parent / "official_training_summary.json").is_file()
    assert "fake official train" in result.paths.console_log.read_text(encoding="utf-8")
    assert "fake official train" in capsys.readouterr().out
    assert result.paths.manifest.is_file()
    assert not (tmp_path / "reports/experiments.csv").exists()


def test_official_guards_and_failed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Official adapter отклоняет чужой pipeline, resume и фиксирует failure."""
    custom_path = _tiny_project(tmp_path / "custom")
    with pytest.raises(ValueError, match="official_reference"):
        run_official_reference(TrainRequest(custom_path, tmp_path / "custom"))

    resume_root = tmp_path / "resume"
    official_path = _official_config(resume_root)
    with pytest.raises(ValueError, match="resume"):
        run_official_reference(
            TrainRequest(
                official_path,
                resume_root,
                resume=True,
                publish_summary=False,
                output_root_override=resume_root / "isolated",
            )
        )

    failure_root = tmp_path / "failure"
    failure_path = _official_config(failure_root)
    monkeypatch.setattr(
        "uzner.training.official.load_official_modules",
        lambda _root: (SimpleNamespace(), SimpleNamespace()),
    )
    monkeypatch.setattr(
        "uzner.training.official.download_official_snapshot",
        lambda _encoder: (_ for _ in ()).throw(RuntimeError("network failed")),
    )
    with pytest.raises(RuntimeError, match="network failed"):
        run_official_reference(
            TrainRequest(
                failure_path,
                failure_root,
                publish_summary=False,
                output_root_override=failure_root / "isolated",
            )
        )
    status_path = failure_root / "isolated/tiny_run/status.json"
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_official_helper_validation(tmp_path: Path) -> None:
    """Адаптер требует один источник и непустую history."""
    config_path = _official_config(tmp_path)
    config = load_experiment_config(config_path)
    data_path = tmp_path / "configs/data.yaml"
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data["data"]["sources"].append({"name": "train2", "path": "data/train.jsonl", "split": "train"})
    data_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="ровно один"):
        _single_source(config, tmp_path, "train")
    with pytest.raises(ValueError, match="history"):
        _epoch_record({}, object())  # type: ignore[arg-type]


def test_official_modules_load_without_project_on_python_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Official kit находится по project_root, как при запуске scripts/*.py."""
    root_string = str(PROJECT_ROOT)
    monkeypatch.setattr(sys, "path", [item for item in sys.path if item != root_string])
    for name in tuple(sys.modules):
        if name == "ner_uz_hackathon_participant" or name.startswith(
            "ner_uz_hackathon_participant."
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)

    predict, train = load_official_modules(PROJECT_ROOT)

    assert predict.__name__.endswith("baseline.predict")
    assert train.__name__.endswith("baseline.train")
    assert root_string not in sys.path


def test_official_snapshot_downloads_only_required_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Загрузка baseline не тянет TensorFlow, Flax и другие лишние веса."""
    (tmp_path / "model.safetensors").write_bytes(b"safe")
    calls: list[dict[str, object]] = []

    def fake_snapshot(*_args: object, **kwargs: object) -> str:
        """Запоминает фильтр файлов и возвращает готовый snapshot."""
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("uzner.training.official_kit.snapshot_download", fake_snapshot)

    result = download_official_snapshot(EncoderConfig("model", "revision"))

    assert result == tmp_path
    assert len(calls) == 1
    patterns = calls[0]["allow_patterns"]
    assert "model.safetensors" in patterns
    assert "pytorch_model.bin" not in patterns
    assert all("tensorflow" not in pattern for pattern in patterns)
