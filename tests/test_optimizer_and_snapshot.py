"""Выбор optimizer и сохранение фактического кода запуска."""

import json
import sys
import tarfile
from types import SimpleNamespace

import torch

from uzner.config import TrainingConfig
from uzner.experiments.source_snapshot import capture_source
from uzner.training.optimizer import make_optimizer


def test_optimizer_selection_preserves_hyperparameters(monkeypatch) -> None:
    """Подставляет GPU optimizer и проверяет явный выбор без изменения AdamW defaults."""

    class FakeAdamW8bit(torch.optim.AdamW):
        """Заменяет CUDA-зависимый optimizer для проверки ветки создания."""

    monkeypatch.setitem(sys.modules, "bitsandbytes.optim", SimpleNamespace(AdamW8bit=FakeAdamW8bit))
    model = torch.nn.Linear(4, 2)
    for name in ("adamw", "adamw_8bit"):
        optimizer = make_optimizer(model, TrainingConfig(optimizer=name))
        assert isinstance(optimizer, FakeAdamW8bit) == (name == "adamw_8bit")
        assert optimizer.defaults["lr"] == 2e-5
        assert optimizer.defaults["weight_decay"] == 0.01
        assert optimizer.defaults["betas"] == (0.9, 0.999)


def test_source_snapshot_excludes_data_and_secrets(tmp_path) -> None:
    """Сохраняет untracked-код, не копируя .env, данные, веса и symlink-и."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "new.py").write_text("print('ok')\n")
    (source / ".env").write_text("SECRET=hidden")
    (source / "data.jsonl").write_text("{}")
    (source / "link.py").symlink_to(source / ".env")
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    environment = tmp_path / "environment"
    environment.mkdir()
    capture_source(tmp_path, environment)
    manifest = json.loads((environment / "source_manifest.json").read_text())
    assert set(manifest) == {"src/new.py", "pyproject.toml"}
    with tarfile.open(environment / "source.tar.gz") as archive:
        assert set(archive.getnames()) == set(manifest)
