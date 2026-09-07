"""TensorRT 10: GPU-буферы PyTorch без промежуточного копирования через CPU."""

import json
from pathlib import Path

import torch

from uzner.data.io import sha256_file


class TensorRTEncoder:
    """Один последовательный execution context на encoder и CUDA stream."""

    def __init__(self, path: Path, precision: str) -> None:
        """Десериализует локальный engine и проверяет контракт тензоров."""
        import tensorrt as trt

        metadata = json.loads(path.with_suffix(".json").read_text("utf-8"))
        if sha256_file(path) != metadata["engine_sha256"]:
            raise ValueError(f"Изменён TensorRT engine: {path}")
        if metadata["precision_flag"] != precision or metadata["tensorrt"] != trt.__version__:
            raise ValueError("TensorRT version/precision не совпадает с engine")
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Не удалось загрузить TensorRT engine: {path}")
        self.context = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        if set(self.names) != {"input_ids", "attention_mask", "hidden"}:
            raise ValueError(f"Неверный контракт engine: {self.names}")
        self.buffers: dict[tuple, torch.Tensor] = {}
        self.stream = torch.cuda.Stream()

    def _dtype(self, name: str) -> torch.dtype:
        """Сопоставляет тип TensorRT типу PyTorch, включая BF16."""
        trt = self.trt
        mapping = {
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.bfloat16: torch.bfloat16,
        }
        return mapping[self.engine.get_tensor_dtype(name)]

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Исполняет engine на текущем stream; владение буферами сохраняет PyTorch."""
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        retained = []
        for name, value in inputs.items():
            tensor = value.to(dtype=self._dtype(name)).contiguous()
            retained.append(tensor)
            if not self.context.set_input_shape(name, tuple(tensor.shape)):
                raise ValueError(f"Форма вне optimization profile: {name} {tensor.shape}")
            self.context.set_tensor_address(name, tensor.data_ptr())
        shape = tuple(self.context.get_tensor_shape("hidden"))
        key = (shape, self._dtype("hidden"))
        # Только последний output: нет неограниченного кэша динамических форм.
        if key not in self.buffers:
            self.buffers.clear()
            self.buffers[key] = torch.empty(shape, device="cuda", dtype=key[1])
        output = self.buffers[key]
        self.context.set_tensor_address("hidden", output.data_ptr())
        current = torch.cuda.current_stream()
        self.stream.wait_stream(current)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 завершился ошибкой")
        for tensor in retained:
            tensor.record_stream(self.stream)
        output.record_stream(self.stream)
        current.wait_stream(self.stream)
        return output
