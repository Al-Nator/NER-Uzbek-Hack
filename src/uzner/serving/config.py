"""Закреплённый состав модели и отдельные настройки исполнения."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Параметры скорости, не меняющие правила выбора сущностей."""

    bundle: Path
    backend: str = "torch"
    precision: str = "bf16"
    window_batch: int = 8
    cpu_threads: int = 1
    crf_cpu: bool = True
    crf_compile: bool = False
    engine_dir: Path | None = None
    engine_models: tuple[str, ...] = ()
    sort_windows: bool = False

    def __post_init__(self) -> None:
        """Проверяет поддерживаемые backend и точность вычислений."""
        if self.backend not in {"torch", "tensorrt"}:
            raise ValueError("Неизвестный backend")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("Неизвестная precision")
        if min(self.window_batch, self.cpu_threads) < 1:
            raise ValueError("Размеры batch и CPU threads должны быть положительными")
        if self.backend == "tensorrt" and (self.engine_dir is None or not self.engine_models):
            raise ValueError("TensorRT требует явные engine_dir и engine_models")
        if len(set(self.engine_models)) != len(self.engine_models) or any(
            name not in {"s33", "s21", "s31"} for name in self.engine_models
        ):
            raise ValueError("Неверный список TensorRT encoder-ов")
        if self.backend == "torch" and self.engine_models:
            raise ValueError("Для torch нельзя задавать TensorRT encoder-ы")
        if self.crf_compile and not self.crf_cpu:
            raise ValueError("Компиляция CRF требует crf_cpu")

    @classmethod
    def read(cls, path: Path) -> "RuntimeConfig":
        """Читает переносимый JSON относительно рабочего каталога сервиса."""
        data = json.loads(path.read_text("utf-8"))
        data["bundle"] = Path(data["bundle"])
        if data.get("engine_dir"):
            data["engine_dir"] = Path(data["engine_dir"])
        if "engine_models" in data:
            data["engine_models"] = tuple(data["engine_models"])
        return cls(**data)
