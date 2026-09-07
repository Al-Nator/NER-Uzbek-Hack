"""Необязательный опрос памяти GPU на узле HTTP-клиента."""

import threading


class MemorySampler:
    """Измеряет NVML device-used, не создавая свой CUDA context."""

    def __init__(self, mode: str = "auto") -> None:
        """Позволяет запускать HTTP benchmark без NVIDIA на клиентской машине."""
        if mode not in {"auto", "required", "off"}:
            raise ValueError("gpu-memory: auto, required или off")
        self.mode = mode
        self.reason: str | None = "disabled" if mode == "off" else None
        self.nvml = None
        self.stop = threading.Event()
        self.values: list[int] = []
        self.utilization: list[int] = []
        self.thread: threading.Thread | None = None
        if mode == "off":
            return
        initialized = False
        try:
            import pynvml

            pynvml.nvmlInit()
            initialized = True
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml = pynvml
        except Exception as error:
            if initialized:
                pynvml.nvmlShutdown()
            if mode == "required":
                raise RuntimeError(f"NVML GPU measurement недоступен: {error}") from error
            self.reason = str(error)

    def _sample(self) -> None:
        """Останавливает опрос при ошибке, не выдавая неполный замер за полный."""
        try:
            while not self.stop.is_set():
                self.values.append(self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used)
                self.utilization.append(self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
                self.stop.wait(0.1)
        except Exception as error:
            self.reason = str(error)

    def start(self) -> None:
        """Запускает фоновый опрос только при доступном NVML."""
        if self.nvml is not None:
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()

    def finish(self) -> dict:
        """Возвращает наблюдаемый пик или явную причину отсутствия замера."""
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
        if not self.values and self.reason is None:
            self.reason = "no NVML samples"
        if self.reason and self.mode == "required":
            raise RuntimeError(f"NVML sampling failed: {self.reason}")
        valid = bool(self.values) and self.reason is None
        return {
            "available": valid,
            "reason": self.reason,
            "measurement_location": "benchmark client host; GPU index 0",
            "sample_interval_ms": 100,
            "samples": len(self.values),
            "device_peak_used_mib": max(self.values) / 2**20 if valid else None,
            "device_min_used_mib": min(self.values) / 2**20 if valid else None,
            "mean_gpu_utilization_percent": sum(self.utilization) / len(self.utilization)
            if valid and self.utilization
            else None,
        }
