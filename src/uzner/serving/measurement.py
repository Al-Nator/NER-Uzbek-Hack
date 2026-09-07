"""Общие измерения latency и точного совпадения ответов, без подбора метрик."""

import platform
import subprocess
from collections.abc import Sequence

from uzner.domain import Prediction


def latency_summary(seconds: Sequence[float]) -> dict[str, float]:
    """Считает процентили по правилу nearest-rank и среднее время запроса."""
    import math

    if not seconds:
        raise ValueError("Нет измерений latency")
    ordered = sorted(seconds)
    result = {
        f"p{q}_ms": ordered[max(0, math.ceil(len(ordered) * q / 100) - 1)] * 1000
        for q in (50, 95, 99)
    }
    return {**result, "mean_ms": sum(seconds) / len(seconds) * 1000, "max_ms": ordered[-1] * 1000}


def compare_predictions(reference: Sequence[Prediction], actual: Sequence[Prediction]) -> dict:
    """Сравнивает только точные spans, не score и не порядок сущностей."""
    before = {p.hash: {(e.label, e.start, e.end) for e in p.entities} for p in reference}
    after = {p.hash: {(e.label, e.start, e.end) for e in p.entities} for p in actual}
    if len(before) != len(reference) or len(after) != len(actual) or before.keys() != after.keys():
        raise ValueError("Для parity требуется одинаковый полный набор уникальных hash")
    changes = [
        {
            "hash": key,
            "added": sorted(after[key] - before[key]),
            "removed": sorted(before[key] - after[key]),
        }
        for key in before
        if before[key] != after[key]
    ]
    return {
        "documents": len(before),
        "changed_documents": len(changes),
        "added_spans": sum(len(row["added"]) for row in changes),
        "removed_spans": sum(len(row["removed"]) for row in changes),
        "changes": changes,
    }


def hardware() -> dict:
    """Сохраняет фактические драйвер, GPU и CPU измерительного узла."""
    result = {
        "measurement_location": "benchmark client host; not inferred remote server hardware",
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch

        result.update(torch=torch.__version__, torch_cuda=torch.version.cuda)
    except ImportError:
        result.update(torch=None, torch_cuda=None)
    commands = {
        "gpu": [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,power.limit",
            "--format=csv,noheader",
        ],
        "cpu": ["lscpu"],
    }
    for name, command in commands.items():
        try:
            result[name] = subprocess.check_output(
                command, text=True, stderr=subprocess.PIPE, timeout=5
            ).strip()
        except (OSError, subprocess.SubprocessError) as error:
            result[name] = "unavailable"
            result[f"{name}_error"] = str(error)
    return result
