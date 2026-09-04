"""Тесты артефактов, логов, отчётов и runtime-метаданных."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.helpers import FakeTokenizer, TinyEncoder, sample_documents
from uzner.config import TokenizationConfig, load_experiment_config
from uzner.data.windows import WindowFeature
from uzner.domain import Prediction
from uzner.evaluation.slices import evaluate_detailed
from uzner.evaluation.tokenizer_audit import AuditSlice, TokenizerAudit
from uzner.experiments.artifacts import (
    finalize_artifact_manifest,
    prepare_run_paths,
    write_jsonl,
)
from uzner.experiments.logging import EpochRecord, RunLogger
from uzner.experiments.reporting import (
    render_training_svg,
    update_comparison_csv,
    write_run_report,
)
from uzner.training.data_setup import collect_data_hashes, load_split, make_loader
from uzner.training.finalization import finalize_outputs, make_epoch_record, write_epoch_metrics
from uzner.training.inference import InferenceResult
from uzner.training.loop import TrainEpochResult
from uzner.training.runtime import (
    capture_environment,
    collect_runtime_info,
    resolve_device,
    set_reproducible_seed,
    validate_context_length,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/e10_xlmr_base_bio.yaml"


def _record(epoch: int = 1) -> EpochRecord:
    """Создаёт плоскую тестовую строку эпохи."""
    return EpochRecord(
        epoch=epoch,
        train_loss=1.0 / epoch,
        dev_loss=0.8 / epoch,
        micro_precision=0.7,
        micro_recall=0.6,
        micro_f1=0.65,
        macro_f1=0.6,
        org_f1=0.5,
        name_f1=0.6,
        geo_f1=0.7,
        learning_rate=2e-5,
        mean_gradient_norm=0.9,
        train_seconds=2.0,
        eval_seconds=1.0,
        train_tokens_per_second=100.0,
        eval_documents_per_second=20.0,
        peak_gpu_memory_gib=1.5,
    )


def _evaluation() -> object:
    """Считает идеальную детальную оценку."""
    documents = sample_documents()
    predictions = tuple(
        Prediction(hash=document.hash, entities=document.entities) for document in documents
    )
    return evaluate_detailed(documents, predictions, documents, {})


def _audit() -> TokenizerAudit:
    """Создаёт компактный tokenizer audit для отчёта."""
    overall = AuditSlice(
        entities=4,
        representable=4,
        representability=1.0,
        mean_subwords=1.0,
        p95_subwords=1.0,
        mean_chars_per_subword=4.0,
    )
    return TokenizerAudit("fake", "rev", 128, 32, 2, 2, 1.0, 1, overall, {})


def test_logger_writes_machine_and_human_logs_and_resumes(tmp_path: Path) -> None:
    """Logger пишет JSONL, CSV, Rich-текст и восстанавливает history."""
    config = load_experiment_config(CONFIG_PATH)
    paths = prepare_run_paths(config, project_root=tmp_path)
    logger = RunLogger(config.run_id, paths)
    evaluation = _evaluation()

    logger.start({"device": "cpu", "encoder": "fake"})
    logger.train_step(
        epoch=1,
        step=3,
        loss=0.4,
        learning_rate=2e-5,
        gradient_norm=None,
        tokens_per_second=50,
        gpu_memory_gib=0,
    )
    logger.epoch(_record(), evaluation, best=True)
    resumed = RunLogger(config.run_id, paths)

    assert resumed.history == (_record(),)
    assert "new best" in paths.console_log.read_text(encoding="utf-8")
    events = [json.loads(line) for line in paths.events.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == [
        "run_started",
        "train_step",
        "epoch_completed",
    ]
    with paths.history.open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 1


def test_reporting_manifest_and_legacy_comparison_migration(tmp_path: Path) -> None:
    """Отчёты и manifest атомарны, а старая CSV мигрирует без падения."""
    config = load_experiment_config(CONFIG_PATH)
    paths = prepare_run_paths(config, project_root=tmp_path)
    evaluation = _evaluation()
    write_jsonl(paths.errors, ({"kind": "missed"}, {"kind": "spurious"}))
    render_training_svg((_record(1), _record(2)), paths.training_plot)
    write_run_report(paths.report, config, _record(), evaluation, _audit())
    comparison = tmp_path / "reports/experiments.csv"
    comparison.parent.mkdir(parents=True)
    comparison.write_text("run_id,old_field\nold,legacy\n", encoding="utf-8")
    update_comparison_csv(comparison, config, _record(), paths.root)
    entries = finalize_artifact_manifest(paths)

    assert "<svg" in paths.training_plot.read_text(encoding="utf-8")
    assert "Boundary-only F1" in paths.report.read_text(encoding="utf-8")
    assert {row["run_id"] for row in csv.DictReader(comparison.open(encoding="utf-8"))} == {
        "old",
        config.run_id,
    }
    assert "\r" not in comparison.read_text(encoding="utf-8")
    assert any(entry.path == "metrics/errors.jsonl" for entry in entries)
    assert json.loads(paths.manifest.read_text(encoding="utf-8"))["schema_version"] == 1
    with pytest.raises(ValueError, match="хотя бы"):
        render_training_svg((), tmp_path / "empty.svg")


def test_finalization_helpers_write_complete_run(tmp_path: Path) -> None:
    """Финализация сохраняет весь набор обязательных артефактов."""
    config = load_experiment_config(CONFIG_PATH)
    paths = prepare_run_paths(config, project_root=tmp_path)
    logger = RunLogger(config.run_id, paths)
    evaluation = _evaluation()
    documents = sample_documents()
    predictions = tuple(Prediction(item.hash, item.entities) for item in documents)
    inference = InferenceResult(predictions, evaluation, 0.4, 1.0, 2.0)
    trained = TrainEpochResult(0.5, 2.0, 100.0, 0.7, 0.0, 4)
    record = make_epoch_record(1, trained, inference, 2e-5)
    logger.epoch(record, evaluation, best=True)

    write_epoch_metrics(paths, 1, inference)
    finalize_outputs(
        paths,
        config,
        record,
        inference,
        _audit(),
        logger,
        tmp_path,
        publish_summary=True,
    )

    assert paths.predictions.is_file()
    assert paths.metrics.is_file()
    assert paths.slice_metrics.is_file()
    assert paths.error_summary.is_file()
    assert paths.tokenizer_audit.is_file()
    assert (paths.metrics.parent / "epochs/epoch_01.json").is_file()
    assert json.loads(paths.status.read_text(encoding="utf-8"))["status"] == "complete"


def test_data_setup_loaders_and_hashes() -> None:
    """Загрузка применяет limit, хэширует данные и собирает batch."""
    config = load_experiment_config(CONFIG_PATH)
    documents = load_split(config, PROJECT_ROOT, "dev", 1)
    hashes = collect_data_hashes(config, PROJECT_ROOT)
    feature = WindowFeature(0, 0, (1, 2), (1, 1), None, ((0, 0), (0, 1)), (-100, 0))
    loader = make_loader(
        (feature,),
        FakeTokenizer(),
        batch_size=1,
        shuffle=False,
        seed=42,
        num_workers=0,
    )

    assert len(documents) == 1
    assert {value["split"] for value in hashes.values()} == {"train", "dev"}
    assert next(iter(loader)).input_ids.shape == (1, 2)
    with pytest.raises(ValueError, match="положительным"):
        load_split(config, PROJECT_ROOT, "dev", 0)
    no_pad = SimpleNamespace(pad_token_id=None)
    with pytest.raises(ValueError, match="pad_token_id"):
        make_loader((feature,), no_pad, batch_size=1, shuffle=False, seed=1, num_workers=0)


def test_runtime_seed_device_context_and_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime фиксирует seed, устройство, Git и lock-файл."""
    set_reproducible_seed(7)
    first = (random.random(), torch.rand(1).item())
    set_reproducible_seed(7)
    assert first == (random.random(), torch.rand(1).item())

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(False).type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device(True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(True).type == "cuda"

    config = load_experiment_config(CONFIG_PATH)
    small = replace(config, tokenization=TokenizationConfig(max_length=64, stride=8))
    validate_context_length(TinyEncoder(max_positions=64), small)
    with pytest.raises(ValueError, match="max_position_embeddings"):
        validate_context_length(TinyEncoder(max_positions=32), small)

    monkeypatch.setattr(
        "uzner.training.runtime._git_value",
        lambda _root, *args: "abc123" if args[0] == "rev-parse" else " M file",
    )
    runtime = collect_runtime_info(tmp_path, torch.device("cpu"))
    assert runtime.git_commit == "abc123"
    assert runtime.git_dirty is True
    assert runtime.to_mapping()["device"] == "cpu"

    (tmp_path / "uv.lock").write_text("lock", encoding="utf-8")
    config_for_paths = replace(config, output_root="runs")
    paths = prepare_run_paths(config_for_paths, project_root=tmp_path)
    capture_environment(paths, tmp_path, runtime)
    assert (paths.environment / "uv.lock").read_text(encoding="utf-8") == "lock"
    assert json.loads((paths.environment / "process.json").read_text(encoding="utf-8"))["pid"] > 0
