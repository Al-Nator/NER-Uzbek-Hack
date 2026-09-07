"""Проверки исходного рецепта s62 и фиксированного low-LR checkpoint."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from tests.test_training_pipeline import _tiny_project
from uzner.config import ExperimentConfig, load_experiment_config
from uzner.training.final_ensemble import STAGES
from uzner.training.final_fit import FinalFitRequest, final_documents, run_final_fit


def test_final_recipes_match_reference() -> None:
    """Encoder/head/окна/батчи остаются исходными, меняется только использование dev."""
    refs = (
        "s32_bge_m3_retromae_global_pointer_a100-continuation-v1",
        "s33_bge_global_pointer_low_lr_a100-low-lr-v1",
        "s21_mdeberta_v3_base_bioes_crf_s2-sequence-v1",
        "s31_xlmr_large_global_pointer_a100-continuation-v1",
    )
    for stage, ref in zip(STAGES, refs, strict=True):
        path = Path("runs") / ref / "resolved_config.yaml"
        if not path.exists():
            pytest.skip("Локальные reference runs отсутствуют")
        source = ExperimentConfig.from_mapping(yaml.safe_load(path.read_text()))
        target = load_experiment_config(stage.config_path)
        expected = replace(source.training, early_stopping_patience=0)
        if source.training.initial_checkpoint:
            expected = replace(
                expected, initial_checkpoint="runs/f10_s32_train_dev_epoch5/checkpoints/best"
            )
        assert target.training == expected
        assert (target.encoder, target.model, target.tokenization) == (
            source.encoder,
            source.model,
            source.tokenization,
        )
        assert len(final_documents(target, Path.cwd())) == 14500
    assert [s.selected_epoch for s in STAGES] == [5, 1, 5, 5]
    assert len([s for s in STAGES if s.ensemble_member]) == 3


def test_final_crf_and_fresh_warm_start(tmp_path: Path, monkeypatch) -> None:
    """CRF работает в final-fit; low-LR сохраняет двухэпоховый scheduler и первую эпоху."""
    path = _tiny_project(tmp_path, epochs=1)
    payload = yaml.safe_load(path.read_text())
    config = payload["experiment"]
    config["model"] = {"head": "crf", "decoder": "crf", "tag_scheme": "bioes"}
    config["training"]["early_stopping_patience"] = 0
    path.write_text(yaml.safe_dump(payload))
    monkeypatch.setattr("uzner.training.final_fit.start_mlflow_run", lambda *a, **k: None)
    first = run_final_fit(FinalFitRequest(path, tmp_path, tmp_path / "smoke"))
    assert (first / "checkpoints/best/head.safetensors").stat().st_ino == (
        first / "checkpoints/last/head.safetensors"
    ).stat().st_ino
    config["run_id"] = "low_lr"
    config["training"].update(
        epochs=2,
        learning_rate=2e-6,
        warmup_ratio=0,
        initial_checkpoint=str(first / "checkpoints/best"),
    )
    path.write_text(yaml.safe_dump(payload))
    second = run_final_fit(FinalFitRequest(path, tmp_path, tmp_path / "smoke", selected_epoch=1))
    metadata = json.loads((second / "metadata.json").read_text())
    assert metadata["scheduler_epochs"] == 2 and metadata["selected_epoch"] == 1
    assert metadata["initial_checkpoint"]["optimizer_state"] == "reset"
    assert not (second / "metrics/dev.json").exists()
    import torch

    state = torch.load(second / "checkpoints/last/training_state.pt", weights_only=False)
    assert state["optimizer"]["param_groups"][0]["lr"] == pytest.approx(1e-6)
    for invalid in (0, 3):
        with pytest.raises(ValueError):
            run_final_fit(FinalFitRequest(path, tmp_path, selected_epoch=invalid))


def test_queue_requires_successful_archive(tmp_path: Path, monkeypatch) -> None:
    """Не запускает новое обучение при ошибке архивирования или неверном отчёте."""
    from types import SimpleNamespace

    from scripts import start_final_ensemble_after_archive as controller

    monkeypatch.setattr(controller, "ARCHIVE", tmp_path)
    monkeypatch.setattr(controller, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(controller.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="обучение не запущено"):
        controller.wait_for_archive()
    report = tmp_path / "deletion_complete.json"
    report.write_text(json.dumps({"verification_passes": 1, "removed_remote_runs": ["s47"]}))
    with pytest.raises(ValueError):
        controller.wait_for_archive()
    report.write_text(json.dumps({"verification_passes": 2, "removed_remote_runs": ["s47"]}))
    controller.wait_for_archive()
