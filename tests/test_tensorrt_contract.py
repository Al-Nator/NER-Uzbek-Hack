"""Контракты TensorRT: SHA, типы, формы, binding и ошибки экспорта без GPU."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from uzner.data.io import sha256_file
from uzner.serving import export
from uzner.serving.tensorrt_runtime import TensorRTEncoder


@pytest.fixture
def trt_fixture(tmp_path, monkeypatch):
    """Подставляет только native TensorRT/CUDA API; Python-контракт остаётся настоящим."""
    context = MagicMock()
    context.get_tensor_shape.return_value = (2, 4, 8)
    context.set_input_shape.return_value = True
    context.execute_async_v3.return_value = True
    engine = MagicMock(num_io_tensors=3)
    engine.get_tensor_name.side_effect = ["input_ids", "attention_mask", "hidden"]
    engine.create_execution_context.return_value = context
    engine.get_tensor_dtype.side_effect = lambda name: "bf16" if name == "hidden" else "i64"
    fake = SimpleNamespace(
        __version__="10.3.0",
        Logger=MagicMock(),
        Runtime=MagicMock(),
        int32="i32",
        int64="i64",
        float32="f32",
        float16="f16",
        bfloat16="bf16",
    )
    fake.Runtime.return_value.deserialize_cuda_engine.return_value = engine
    monkeypatch.setitem(sys.modules, "tensorrt", fake)
    stream, current = MagicMock(), MagicMock()
    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current)
    path = tmp_path / "s33.plan"
    path.write_bytes(b"verified engine")
    metadata = {"engine_sha256": sha256_file(path), "precision_flag": "bf16", "tensorrt": "10.3.0"}
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return SimpleNamespace(
        path=path,
        trt=fake,
        engine=engine,
        context=context,
        stream=stream,
        current=current,
        metadata=metadata,
    )


@pytest.mark.parametrize("fault", ["hash", "version", "precision", "deserialize", "names"])
def test_engine_rejects_corrupt_or_incompatible_artifact(trt_fixture, fault):
    """Не использует изменённые планы или другой TensorRT/precision и не делает fallback."""
    item = trt_fixture
    if fault == "hash":
        item.path.write_bytes(b"modified")
    elif fault == "version":
        item.trt.__version__ = "10.4.0"
    elif fault == "precision":
        item.metadata["precision_flag"] = "fp16"
        item.path.with_suffix(".json").write_text(json.dumps(item.metadata))
    elif fault == "deserialize":
        item.trt.Runtime.return_value.deserialize_cuda_engine.return_value = None
    else:
        item.engine.get_tensor_name.side_effect = ["wrong", "attention_mask", "hidden"]
    with pytest.raises((ValueError, RuntimeError)):
        TensorRTEncoder(item.path, "bf16")


class FakeTensor:
    """Имитирует CUDA-буфер, фиксируя dtype и stream ownership."""

    def __init__(self, shape):
        """Сохраняет форму и список stream-зависимостей."""
        self.shape = shape
        self.streams = []

    def to(self, *, dtype):
        """Фиксирует тип входного TensorRT binding."""
        self.dtype = dtype
        return self

    def contiguous(self):
        """Возвращает плотно размещённый тестовый буфер."""
        return self

    def data_ptr(self):
        """Возвращает различимый адрес буфера без настоящего GPU allocation."""
        return id(self)

    def record_stream(self, stream):
        """Удерживает входной буфер до завершения execution stream."""
        self.streams.append(stream)


def test_binding_types_shapes_streams_and_bounded_output_cache(trt_fixture, monkeypatch):
    """Проверяет прямые адреса, CUDA stream ordering и отсутствие растущего кэша форм."""
    item = trt_fixture
    allocations = []

    def allocate(shape, *, device, dtype):
        """Проверяет место и тип выделения выходного hidden tensor."""
        assert device == "cuda" and dtype == torch.bfloat16
        tensor = FakeTensor(shape)
        allocations.append(tensor)
        return tensor

    monkeypatch.setattr(torch, "empty", allocate)
    runtime = TensorRTEncoder(item.path, "bf16")
    ids, mask = FakeTensor((2, 4)), FakeTensor((2, 4))
    first = runtime(ids, mask)
    assert ids.dtype == mask.dtype == torch.int64
    assert ids.streams == mask.streams == first.streams == [item.stream]
    item.context.set_tensor_address.assert_any_call("input_ids", ids.data_ptr())
    item.context.set_tensor_address.assert_any_call("hidden", first.data_ptr())
    item.stream.wait_stream.assert_called_with(item.current)
    item.current.wait_stream.assert_called_with(item.stream)
    assert runtime(ids, mask) is first
    assert len(allocations) == 1
    item.context.get_tensor_shape.return_value = (1, 3, 8)
    assert runtime(ids, mask) is not first
    assert len(runtime.buffers) == 1 and len(allocations) == 2
    for native, expected in [("i32", torch.int32), ("f16", torch.float16), ("f32", torch.float32)]:
        item.engine.get_tensor_dtype.side_effect = lambda name, value=native: value
        assert runtime._dtype("hidden") == expected


@pytest.mark.parametrize("shape_ok", [False, True])
def test_bad_shape_or_native_execution_is_fatal(trt_fixture, monkeypatch, shape_ok):
    """Не маскирует выход за profile или отказ native execution пустым ответом."""
    item = trt_fixture
    item.context.set_input_shape.return_value = shape_ok
    item.context.execute_async_v3.return_value = False
    monkeypatch.setattr(torch, "empty", lambda shape, **kwargs: FakeTensor(shape))
    runtime = TensorRTEncoder(item.path, "bf16")
    with pytest.raises((ValueError, RuntimeError)):
        runtime(FakeTensor((1, 4)), FakeTensor((1, 4)))


def test_export_refuses_cpu_and_reuses_existing_onnx(tmp_path, monkeypatch):
    """CPU не становится скрытым export fallback; готовый ONNX не затирается."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        export.export_encoder(tmp_path, tmp_path / "out")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    output = tmp_path / "out"
    output.mkdir()
    (output / "encoder.onnx").write_bytes(b"original")
    assert export.export_encoder(tmp_path, output).read_bytes() == b"original"


def test_encoder_graph_returns_sequence_without_pooler():
    """Экспортируемый граф выдаёт last hidden state, а не pooled embedding."""
    encoder = MagicMock()
    encoder.eval.return_value = encoder
    encoder.return_value = (torch.ones(1, 2, 3), torch.zeros(1, 3))
    graph = export.EncoderGraph(encoder)
    ids = torch.ones(1, 2, dtype=torch.long)
    assert graph(ids, ids).shape == (1, 2, 3)
    assert encoder.call_args.kwargs["return_dict"] is False


def test_export_keeps_dynamic_axes_and_offline_loading(tmp_path, monkeypatch):
    """Проверяет параметры настоящего export-wrapper, не создавая native CUDA-граф."""
    encoder = MagicMock()
    encoder.eval.return_value = encoder
    loaded = []

    def load(path, **kwargs):
        """Фиксирует local-only чтение encoder-а."""
        loaded.append((path, kwargs))
        return encoder

    def save(graph, args, target, **kwargs):
        """Проверяет dynamic batch/length и фиксированный opset."""
        assert args[0].shape == (1, 32)
        assert args[0][0, 0] == 5
        assert kwargs["opset_version"] == 17
        assert kwargs["dynamic_axes"]["hidden"] == {0: "batch", 1: "length"}
        assert kwargs["external_data"] and not kwargs["dynamo"]
        from pathlib import Path

        Path(target).write_bytes(b"exported")

    original_ones = torch.ones
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(export.AutoModel, "from_pretrained", load)
    monkeypatch.setattr(export.EncoderGraph, "cuda", lambda self: self)
    monkeypatch.setattr(
        torch, "ones", lambda shape, dtype, device: original_ones(shape, dtype=dtype)
    )
    monkeypatch.setattr(torch.onnx, "export", save)
    output = export.export_encoder(tmp_path / "checkpoint", tmp_path / "export")
    assert output.read_bytes() == b"exported"
    assert loaded[0][1] == {"local_files_only": True, "attn_implementation": "eager"}


@pytest.fixture
def builder_fixture(tmp_path, monkeypatch):
    """Подменяет TensorRT builder, оставляя проверку metadata и profile в production-коде."""
    network = MagicMock(num_inputs=2)
    network.get_input.side_effect = [
        SimpleNamespace(name="input_ids"),
        SimpleNamespace(name="attention_mask"),
    ]
    builder, parser = MagicMock(), MagicMock()
    builder.create_network.return_value = network
    builder.build_serialized_network.return_value = b"new engine"
    parser.parse_from_file.return_value = True
    fake = SimpleNamespace(
        __version__="10.3.0",
        Logger=MagicMock(),
        Builder=MagicMock(return_value=builder),
        OnnxParser=MagicMock(return_value=parser),
        NetworkDefinitionCreationFlag=SimpleNamespace(EXPLICIT_BATCH=0),
        MemoryPoolType=SimpleNamespace(WORKSPACE=0),
        BuilderFlag=SimpleNamespace(TF32="tf32", BF16="bf16", FP16="fp16"),
    )
    monkeypatch.setitem(sys.modules, "tensorrt", fake)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "mock A100")
    onnx = tmp_path / "encoder.onnx"
    onnx.write_bytes(b"onnx")
    return SimpleNamespace(
        onnx=onnx, output=tmp_path / "engine.plan", builder=builder, parser=parser
    )


@pytest.mark.parametrize("precision", ["bf16", "fp16", "fp32"])
def test_build_engine_records_precision_hashes_and_fixed_profile(builder_fixture, precision):
    """Не меняет shape profile и фиксирует SHA/precision рядом с каждым plan."""
    item = builder_fixture
    result = export.build_engine(item.onnx, item.output, precision)
    metadata = json.loads(result.with_suffix(".json").read_text())
    assert metadata["engine_sha256"] == sha256_file(result)
    assert metadata["onnx_sha256"] == sha256_file(item.onnx)
    assert metadata["precision_flag"] == precision
    assert metadata["profile"]["max"] == [16, 512]
    profile = item.builder.create_optimization_profile.return_value
    profile.set_shape.assert_any_call("input_ids", (1, 2), (8, 128), (16, 512))
    with pytest.raises(FileExistsError):
        export.build_engine(item.onnx, item.output, precision)


@pytest.mark.parametrize("fault", ["parse", "build", "precision"])
def test_failed_build_does_not_publish_plan(builder_fixture, fault):
    """Отклоняет native-ошибки и неизвестную точность до публикации артефакта."""
    item = builder_fixture
    if fault == "parse":
        item.parser.parse_from_file.return_value = False
        item.parser.num_errors = 1
        item.parser.get_error.return_value = "bad ONNX"
    if fault == "build":
        item.builder.build_serialized_network.return_value = None
    with pytest.raises((ValueError, RuntimeError)):
        export.build_engine(item.onnx, item.output, "bad" if fault == "precision" else "bf16")
    assert not item.output.exists()
