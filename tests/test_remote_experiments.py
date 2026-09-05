from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

from uzner.config import load_experiment_config
from uzner.data.io import sha256_file
from uzner.experiments.remote import (
    RemoteWorkspace,
    remote_status,
    selected_run_ids,
    ssh_command,
    start_remote_training,
)
from uzner.experiments.remote_sync import copy_remote_run, remote_run_status, verify_run_manifest

PROJECT_ROOT = Path(__file__).parents[1]


def _workspace(root: Path = PROJECT_ROOT) -> RemoteWorkspace:
    """Создаёт тестовое описание remote workspace."""
    return RemoteWorkspace(
        local_root=root,
        host="gpu-host",
        remote_root=PurePosixPath("/srv/ner project"),
    )


def test_second_series_remote_selection_has_only_s22_and_s23() -> None:
    """Этап xlmr-large разрешает ровно два A100 run ID."""
    run_ids = selected_run_ids(
        _workspace(),
        Path("configs/series/second_a100.yaml"),
        "xlmr-large",
        "a100-v2",
    )

    assert run_ids == (
        "s22_xlmr_large_bioes_constrained_a100-v2",
        "s23_xlmr_large_bioes_crf_a100-v2",
    )


@pytest.mark.parametrize(
    "name",
    ["s22_xlmr_large_bioes_constrained.yaml", "s23_xlmr_large_bioes_crf.yaml"],
)
def test_a100_configs_keep_effective_batch_without_checkpointing(name: str) -> None:
    """Оба A100 run-а сохраняют effective batch и отключают checkpointing."""
    config = load_experiment_config(PROJECT_ROOT / "configs" / "experiments" / "a100" / name)

    assert config.training.batch_size == 8
    assert config.training.gradient_accumulation_steps == 1
    assert config.training.eval_batch_size == 16
    assert config.training.gradient_checkpointing is False


def test_ssh_command_quotes_remote_arguments() -> None:
    """Путь с пробелом передаётся одним SSH-аргументом."""
    command = ssh_command(_workspace(), ["cat", "/srv/ner project/status.json"])

    assert command[-2] == "gpu-host"
    assert command[-1] == "cat '/srv/ner project/status.json'"


def test_training_command_uses_remote_mlflow_and_detached_zellij(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Запуск пишет в remote experiment и не зависит от SSH."""
    workspace = _workspace()
    calls: list[list[str]] = []
    monkeypatch.setattr("uzner.experiments.remote._session_names", lambda _workspace: set())
    monkeypatch.setattr(
        "uzner.experiments.remote._run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": 1})(),
    )

    def fake_ssh(_workspace: RemoteWorkspace, command: list[str], **_kwargs: object) -> str:
        """Сохраняет удалённую команду без запуска."""
        calls.append(command)
        return ""

    monkeypatch.setattr("uzner.experiments.remote._ssh", fake_ssh)

    run_ids = start_remote_training(
        workspace,
        series_path=Path("configs/series/second_a100.yaml"),
        stage="xlmr-large",
        suffix="a100-v2",
        session="uzner-s22-s23",
    )

    assert len(run_ids) == 2
    assert calls[0] == ["zellij", "attach", "--create-background", "uzner-s22-s23"]
    training = calls[1]
    assert "UZNER_MLFLOW_EXPERIMENT=uzner-second-series-a100" in training
    assert training[-6:] == [
        "--series",
        "configs/series/second_a100.yaml",
        "--stage",
        "xlmr-large",
        "--run-suffix",
        "a100-v2",
    ]


def test_manifest_verification_detects_corruption(tmp_path: Path) -> None:
    """Манифест подтверждает байты и SHA-256 после rsync."""
    artifact = tmp_path / "metrics" / "dev.json"
    artifact.parent.mkdir()
    artifact.write_text('{"f1": 0.9}\n', encoding="utf-8")
    manifest = {
        "artifacts": [
            {
                "path": "metrics/dev.json",
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        ]
    }
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    verify_run_manifest(tmp_path)
    artifact.write_text("broken\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Не совпал артефакт"):
        verify_run_manifest(tmp_path)


def test_remote_status_is_parsed_and_run_id_is_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Статус читается как JSON, а небезопасное имя отклоняется."""
    completed = type(
        "Result",
        (),
        {"returncode": 0, "stdout": '{"status": "complete"}\n'},
    )()
    monkeypatch.setattr("uzner.experiments.remote_sync._run", lambda *_args, **_kwargs: completed)

    assert remote_run_status(_workspace(), "s22_a100") == "complete"
    with pytest.raises(ValueError, match="Небезопасный run_id"):
        remote_run_status(_workspace(), "../escape")


def test_remote_status_compacts_zellij_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Статус не выводит весь технический JSON zellij."""
    panes = json.dumps(
        [
            {"is_plugin": True, "title": "status-bar"},
            {
                "is_plugin": False,
                "title": "training",
                "exited": False,
                "terminal_command": "uv run python scripts/run_series.py",
                "pane_command": None,
            },
        ]
    )
    outputs = iter([panes, "NVIDIA A100, 97 %, 1234 MiB, 81920 MiB"])
    monkeypatch.setattr(
        "uzner.experiments.remote._ssh",
        lambda *_args, **_kwargs: next(outputs),
    )

    status = remote_status(_workspace(), "training")

    assert "training (running): uv run python scripts/run_series.py" in status
    assert "status-bar" not in status


def test_metrics_first_then_checkpoint_completion(tmp_path: Path, monkeypatch) -> None:
    """Лёгкий импорт видим сразу, полный sync догружает и проверяет веса."""
    import shutil

    remote = tmp_path / "remote"
    (remote / "metrics").mkdir(parents=True)
    (remote / "checkpoints").mkdir()
    (remote / "metrics/dev.json").write_text("{}")
    (remote / "checkpoints/weights.bin").write_bytes(b"weights")
    manifest = {
        "artifacts": [
            {
                "path": str(path.relative_to(remote)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in remote.rglob("*")
            if path.is_file()
        ]
    }
    (remote / "artifact_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr("uzner.experiments.remote_sync.remote_run_status", lambda *_: "complete")

    def fake_rsync(command):
        """Копирует fixture вместо реального SSH и сохраняет exclude-семантику."""
        source = remote if command[-2].startswith("gpu-host:") else Path(command[-2])
        ignore = (
            shutil.ignore_patterns("checkpoints") if "--exclude=/checkpoints/" in command else None
        )
        shutil.copytree(source, command[-1], dirs_exist_ok=True, ignore=ignore)

    monkeypatch.setattr("uzner.experiments.remote_sync._run", fake_rsync)
    workspace = _workspace(tmp_path)
    target, copied = copy_remote_run(workspace, "s22", include_checkpoints=False)
    assert copied
    assert (target / ".checkpoint-transfer-pending").exists()
    assert not (target / "checkpoints").exists()
    verify_run_manifest(target, include_checkpoints=False)
    with pytest.raises(FileNotFoundError):
        verify_run_manifest(target)
    assert copy_remote_run(workspace, "s22", include_checkpoints=False) == (target, False)
    assert copy_remote_run(workspace, "s22") == (target, True)
    assert not (target / ".checkpoint-transfer-pending").exists()
    verify_run_manifest(target)
    assert copy_remote_run(workspace, "s22") == (target, False)
