"""Проверки зависимости очереди без реальных GPU-процессов и ожидания."""

import pytest

from uzner.experiments.deferred import ProcessIdentity


def test_process_identity(tmp_path):
    """Не путает завершённый процесс, zombie и повторно выданный PID."""
    process = ProcessIdentity.parse("123:456")
    assert not process.alive(tmp_path)
    folder = tmp_path / "123"
    folder.mkdir()
    fields = ["S"] + ["0"] * 18 + ["456"]
    stat = folder / "stat"
    stat.write_text("123 (name with ) parens) " + " ".join(fields))
    assert process.alive(tmp_path)
    fields[0] = "Z"
    stat.write_text("123 (name) " + " ".join(fields))
    assert not process.alive(tmp_path)
    fields[0], fields[19] = "S", "789"
    stat.write_text("123 (name) " + " ".join(fields))
    assert not process.alive(tmp_path)


@pytest.mark.parametrize("value", ["0:1", "-1:1", "123:x", "1", "1:2:3"])
def test_bad_identity(value):
    """Не допускает неоднозначных и некорректных идентификаторов ожидания."""
    with pytest.raises(ValueError):
        ProcessIdentity.parse(value)


def test_deferred_cli_continues_independent_failures(tmp_path, monkeypatch):
    """После зависимости запускает каждый конфиг и не скрывает неудачный run."""
    import json
    import runpy
    import subprocess
    import sys
    from types import SimpleNamespace

    state = tmp_path / "queue.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_deferred_series.py",
            "--wait-for",
            "123:456",
            "--series",
            "configs/series/followup_serial_a100.yaml",
            "--run-suffix",
            "test-only",
            "--queue-state",
            str(state),
        ],
    )
    monkeypatch.setattr(ProcessIdentity, "alive", lambda self: False)
    calls = []

    def fake_run(command, *, check):
        """Первый запуск падает; остальные независимые запуски успешны."""
        calls.append(command)
        return SimpleNamespace(returncode=int(len(calls) == 1))

    monkeypatch.setattr(subprocess, "run", fake_run)
    main = runpy.run_path("scripts/run_deferred_series.py")["main"]
    assert main() == 1
    result = json.loads(state.read_text())
    assert result["status"] == "failed" and len(result["results"]) == 3
    assert len(calls) == 3 and all(command[0] == "uv" for command in calls)
    with pytest.raises(FileExistsError):
        main()
