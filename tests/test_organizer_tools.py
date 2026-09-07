"""Проверки файловой оценки, команд Makefile и измерительного клиента."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from scripts import benchmark_service
from uzner.evaluation.files import evaluate_files
from uzner.serving.measurement import hardware
from uzner.serving.memory import MemorySampler

ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path, rows):
    """Пишет маленький JSONL, не меняя оригинальные датасеты."""
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_evaluate_unicode_exact_and_protected_output(tmp_path):
    """Считает точные Unicode-границы, не требует GPU и не затирает отчёт."""
    entity = {"label": "NAME", "start": 2, "end": 5}
    gold = _jsonl(tmp_path / "gold", [{"hash": "x", "text": "🙂 Али", "entities": [entity]}])
    predictions = _jsonl(tmp_path / "pred", [{"hash": "x", "entities": [entity]}])
    output = tmp_path / "metrics.json"
    assert evaluate_files(gold, predictions, output).micro.f1 == 1.0
    assert json.loads(output.read_text())["records"] == 1
    with pytest.raises(FileExistsError):
        evaluate_files(gold, predictions, output)


@pytest.mark.parametrize(
    "rows",
    [
        [{"hash": "x"}],
        [{"hash": "x", "entities": {}}],
        [{"hash": "y", "entities": []}],
        [{"hash": "x", "entities": [{"label": "NAME", "start": 0, "end": 9}]}],
        [],
    ],
)
def test_eval_rejects_invalid_predictions(tmp_path, rows):
    """Не превращает битый файл, потерянные строки и offsets в правдоподобную метрику."""
    gold = _jsonl(tmp_path / "gold", [{"hash": "x", "text": "Ali", "entities": []}])
    predictions = _jsonl(tmp_path / "pred", rows)
    with pytest.raises(ValueError):
        evaluate_files(gold, predictions, tmp_path / "metrics")
    assert not (tmp_path / "metrics").exists()


@pytest.mark.parametrize("entities", [None, {}])
def test_public_input_is_not_gold(tmp_path, entities):
    """Запрещает случайно оценивать public inputs как полностью пустую разметку."""
    row = {"hash": "x", "text": "Ali"}
    if entities is not None:
        row["entities"] = entities
    gold = _jsonl(tmp_path / "gold", [row])
    predictions = _jsonl(tmp_path / "pred", [{"hash": "x", "entities": []}])
    with pytest.raises(ValueError, match="entities"):
        evaluate_files(gold, predictions)


def test_memory_disabled_and_missing_nvml(monkeypatch):
    """CPU-клиент остаётся работоспособен, а отсутствие замера явно отражено в JSON."""
    disabled = MemorySampler("off")
    disabled.start()
    assert disabled.finish()["reason"] == "disabled"
    monkeypatch.setitem(sys.modules, "pynvml", None)
    sampler = MemorySampler()
    sampler.start()
    result = sampler.finish()
    assert result["device_peak_used_mib"] is None
    assert not result["available"]
    with pytest.raises(RuntimeError, match="NVML"):
        MemorySampler("required")
    with pytest.raises(ValueError):
        MemorySampler("wrong")


def test_memory_samples_and_nvml_cleanup(monkeypatch):
    """Считает наблюдаемый device-used peak и закрывает NVML ровно один раз."""
    shutdowns = []
    fake = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=200 * 2**20),
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=50),
        nvmlShutdown=lambda: shutdowns.append(True),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    sampler = MemorySampler("required")
    monkeypatch.setattr(sampler.stop, "wait", lambda seconds: sampler.stop.set())
    sampler.start()
    sampler.thread.join(timeout=2)
    result = sampler.finish()
    assert result["available"]
    assert result["device_peak_used_mib"] == 200
    assert result["mean_gpu_utilization_percent"] == 50
    assert shutdowns == [True]


def test_memory_sampling_failure_is_not_valid_peak(monkeypatch):
    """Отказ NVML после инициализации не выдаёт частичную выборку за успешный замер."""

    def fail(handle):
        """Имитирует потерю драйвера во время опроса."""
        raise RuntimeError("driver lost")

    fake = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetMemoryInfo=fail,
        nvmlShutdown=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    sampler = MemorySampler("required")
    sampler.start()
    sampler.thread.join(timeout=2)
    with pytest.raises(RuntimeError, match="driver lost"):
        sampler.finish()


def test_required_memory_rejects_zero_samples(monkeypatch):
    """Обязательный замер не считается успешным, если поток не снял ни одной пробы."""
    fake = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlShutdown=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    sampler = MemorySampler("required")
    with pytest.raises(RuntimeError, match="no NVML samples"):
        sampler.finish()


def test_hardware_missing_host_utilities(monkeypatch):
    """Отсутствие nvidia-smi и lscpu не роняет HTTP benchmark на клиенте."""

    def missing(*args, **kwargs):
        """Имитирует машину без Linux/NVIDIA CLI."""
        raise FileNotFoundError("utility missing")

    monkeypatch.setattr("uzner.serving.measurement.subprocess.check_output", missing)
    result = hardware()
    assert result["gpu"] == result["cpu"] == "unavailable"
    assert "client host" in result["measurement_location"]


def test_full_benchmark_mocked_http_and_reference(tmp_path, monkeypatch):
    """Прогоняет весь CLI: warmup, два раунда, strict parity, gold и CPU-only отчёт."""
    source = _jsonl(
        tmp_path / "input", [{"hash": str(i), "text": "Ali", "entities": []} for i in range(9)]
    )
    reference = _jsonl(tmp_path / "reference", [{"hash": str(i), "entities": []} for i in range(9)])
    config = tmp_path / "config.json"
    config.write_text("{}")
    client_class = httpx.Client
    sizes = []

    def respond(incoming):
        """Возвращает contract-совместимый ответ и проверяет отсутствие gold."""
        if incoming.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        rows = json.loads(incoming.content)
        assert all(set(row) == {"hash", "text"} for row in rows)
        sizes.append(len(rows))
        return httpx.Response(
            200, json={"data": [{"hash": row["hash"], "entities": []} for row in rows]}
        )

    monkeypatch.setattr(
        benchmark_service.httpx,
        "Client",
        lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setattr(benchmark_service, "hardware", lambda: {"test": True})
    args = [
        "--input",
        str(source),
        "--config",
        str(config),
        "--output",
        str(tmp_path / "out"),
        "--gpu-memory",
        "off",
        "--gold",
        "--reference",
        str(reference),
    ]
    benchmark_service.main(args)
    result = json.loads((tmp_path / "out/summary.json").read_text())
    assert sizes == [8, 8, 8, 8, 1, 8, 1]
    assert result["messages"] == 18
    assert result["requests"] == 4
    assert result["quality"]["records"] == 9
    assert result["parity"]["changed_documents"] == 0
    assert result["gpu_memory"]["device_peak_used_mib"] is None
    with pytest.raises(FileExistsError):
        benchmark_service.main(args)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_benchmark_invalid_rate_stops_before_network(tmp_path, value):
    """Отрицательная, бесконечная или NaN нагрузка не запускает измерение."""
    with pytest.raises(SystemExit):
        benchmark_service.main(
            [
                "--input",
                "missing",
                "--config",
                "missing",
                "--output",
                str(tmp_path / "out"),
                "--arrival-rate",
                value,
            ]
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "name", ["predict_service.py", "benchmark_service.py", "train.py", "evaluate.py"]
)
def test_documented_cli_help_from_another_directory(tmp_path, name):
    """Документированные entrypoints импортируются из установленного uv-пакета."""
    environment = os.environ.copy()
    if "COV_CORE_SOURCE" in environment:
        environment["COV_CORE_CONFIG"] = str(ROOT / "pyproject.toml")
        environment["COV_CORE_BRANCH"] = "enabled"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("target", ["predict", "eval", "benchmark", "train-smoke"])
def test_make_requires_explicit_io_paths(target):
    """Опечатка в команде не запускает модель и не пишет в случайное место."""
    result = subprocess.run(["make", target], cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "Нуж" in result.stdout
