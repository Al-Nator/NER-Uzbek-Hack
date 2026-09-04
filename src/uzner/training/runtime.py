"""Воспроизводимость, устройство и environment-метаданные."""

from __future__ import annotations

import os
import platform
import random
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import transformers

from uzner.config import ExperimentConfig
from uzner.experiments.artifacts import RunPaths, write_json


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    """Снимок окружения и устройства одного run-а."""

    python: str
    torch: str
    transformers: str
    cuda_runtime: str | None
    cudnn: int | None
    device: str
    gpu_name: str | None
    gpu_memory_bytes: int | None
    hostname: str
    git_commit: str | None
    git_dirty: bool | None

    def to_mapping(self) -> dict[str, object]:
        """Сериализует снимок окружения."""
        return asdict(self)


def set_reproducible_seed(seed: int) -> None:
    """Фиксирует Python, CPU и CUDA-генераторы."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(require_gpu: bool) -> torch.device:
    """Выбирает CUDA или явно разрешённый CPU smoke-режим."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_gpu:
        raise RuntimeError("Experiment требует CUDA, но torch.cuda.is_available() == False")
    return torch.device("cpu")


def validate_context_length(model: torch.nn.Module, config: ExperimentConfig) -> None:
    """Запрещает молча превышать нативный контекст encoder-а."""
    model_limit = getattr(model.config, "max_position_embeddings", None)
    requested = config.tokenization.max_length
    if isinstance(model_limit, int) and requested > model_limit:
        raise ValueError(f"max_length={requested} превышает max_position_embeddings={model_limit}")


def _git_value(project_root: Path, *arguments: str) -> str | None:
    """Читает одно безопасное значение Git или возвращает None."""
    completed = subprocess.run(
        ("git", *arguments),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def collect_runtime_info(project_root: Path, device: torch.device) -> RuntimeInfo:
    """Собирает версии, Git и точную GPU без сетевых вызовов."""
    commit = _git_value(project_root, "rev-parse", "HEAD")
    status = _git_value(project_root, "status", "--porcelain")
    has_cuda = device.type == "cuda"
    properties = torch.cuda.get_device_properties(device) if has_cuda else None
    return RuntimeInfo(
        python=platform.python_version(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        cuda_runtime=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        device=str(device),
        gpu_name=None if properties is None else properties.name,
        gpu_memory_bytes=None if properties is None else properties.total_memory,
        hostname=platform.node(),
        git_commit=commit,
        git_dirty=None if status is None else bool(status),
    )


def capture_environment(
    paths: RunPaths,
    project_root: Path,
    runtime: RuntimeInfo,
) -> None:
    """Сохраняет lock-файл и runtime JSON внутри run-артефактов."""
    lock_path = project_root / "uv.lock"
    if lock_path.is_file():
        shutil.copy2(lock_path, paths.environment / "uv.lock")
    write_json(paths.environment / "runtime.json", runtime.to_mapping())
    write_json(
        paths.environment / "process.json",
        {
            "pid": os.getpid(),
            "working_directory": str(project_root),
        },
    )
