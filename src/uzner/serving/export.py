"""Экспорт encoder-а с неизменёнными весами в ONNX и TensorRT 10.3."""

from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from transformers import AutoModel

from uzner.data.io import sha256_file
from uzner.experiments.artifacts import write_json
from uzner.serving.deberta_export import export_self_attention


class EncoderGraph(nn.Module):
    """Отделяет encoder от точных postprocessing и CRF, остающихся общими."""

    def __init__(self, encoder: nn.Module) -> None:
        """Сохраняет encoder с явно выключенным dropout."""
        super().__init__()
        self.encoder = encoder.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Возвращает последнюю последовательность hidden states без pooler."""
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[
            0
        ]


def export_encoder(checkpoint: Path, output: Path) -> Path:
    """Экспортирует динамические batch/length на GPU без доступа к сети."""
    if not torch.cuda.is_available():
        raise RuntimeError("Экспорт проверяется только на CUDA GPU")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "encoder.onnx"
    if target.exists():
        return target
    encoder = AutoModel.from_pretrained(
        checkpoint / "encoder", local_files_only=True, attn_implementation="eager"
    )
    model = EncoderGraph(encoder).cuda().eval().requires_grad_(False)
    ids = torch.ones((1, 32), dtype=torch.long, device="cuda") * 5
    mask = torch.ones_like(ids)
    with torch.inference_mode(), export_self_attention(encoder):
        torch.onnx.export(
            model,
            (ids, mask),
            str(target),
            opset_version=17,
            input_names=["input_ids", "attention_mask"],
            output_names=["hidden"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "length"},
                "attention_mask": {0: "batch", 1: "length"},
                "hidden": {0: "batch", 1: "length"},
            },
            do_constant_folding=True,
            dynamo=False,
            external_data=True,
        )
    del model, encoder
    torch.cuda.empty_cache()
    return target


def build_engine(onnx_path: Path, output: Path, precision: str) -> Path:
    """Создаёт engine с profile 1–16 окон по 2–512 токенов и сохраняет условия."""
    import tensorrt as trt

    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path.resolve())):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)
    if precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
    elif precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision != "fp32":
        raise ValueError("Поддерживаются fp32, bf16 и fp16")
    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        name = network.get_input(index).name
        profile.set_shape(name, (1, 2), (8, 128), (16, 512))
    config.add_optimization_profile(profile)
    config.builder_optimization_level = 3
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT не собрал engine")
    output.write_bytes(bytes(serialized))
    write_json(
        output.with_suffix(".json"),
        {
            "tensorrt": trt.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "precision_flag": precision,
            "build_seconds": perf_counter() - started,
            "engine_sha256": sha256_file(output),
            "onnx_sha256": sha256_file(onnx_path),
            "profile": {"min": [1, 2], "opt": [8, 128], "max": [16, 512]},
            "note": "Builder flag permits precision; not every layer necessarily uses it.",
        },
    )
    return output
