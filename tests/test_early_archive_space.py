"""Защита раннего освобождения места от удаления непроверенных данных."""

import pytest

from scripts import free_verified_a100_space as module


def test_mismatch_never_deletes(monkeypatch, tmp_path):
    """Несовпадение SHA запрещает вызов удаляющего процесса."""
    monkeypatch.setattr(module, "ARCHIVE", tmp_path)
    monkeypatch.setattr(module, "NAMES", ("s32_test",))
    monkeypatch.setattr(module, "remote_scan", lambda name: {"weights": [1, "a"]})
    monkeypatch.setattr(module, "inventory", lambda root: {"weights": [1, "b"]})
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="Архив не совпадает"):
        module.main()
    assert not calls
    assert not (tmp_path / "early_space_ready.json").exists()
