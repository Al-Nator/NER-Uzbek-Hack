"""GPU-preflight pinned encoder-ов выбранной серии без experiment run-ов."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from uzner.config import load_experiment_config
from uzner.experiments.series import load_series
from uzner.models.pretrained import resolve_pretrained_snapshot
from uzner.models.token_tagger import TokenTagger
from uzner.training.runtime import resolve_device, validate_context_length


def parse_args() -> argparse.Namespace:
    """Разбирает манифест, этап и корень проекта."""
    parser = argparse.ArgumentParser(description="Проверить pinned encoder-ы на GPU")
    parser.add_argument("--series", type=Path, default=Path("configs/series/first.yaml"))
    parser.add_argument("--stage", default="encoders")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def _parameter_count(model: torch.nn.Module) -> int:
    """Считает все параметры encoder-а и NER-головы."""
    return sum(parameter.numel() for parameter in model.parameters())


@torch.inference_mode()
def _check_one(config_path: Path, device: torch.device) -> dict[str, object]:
    """Загружает pinned model и выполняет один BF16 forward."""
    config = load_experiment_config(config_path)
    if config.pipeline != "uzner":
        raise ValueError("Preflight поддерживает только pipeline=uzner")
    snapshot = resolve_pretrained_snapshot(config.encoder)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot.path,
        trust_remote_code=config.encoder.trust_remote_code,
        use_fast=True,
        local_files_only=True,
        fix_mistral_regex=False,
    )
    if not tokenizer.is_fast:
        raise ValueError(f"{config.encoder.name}: нужен fast tokenizer")
    model = (
        TokenTagger.from_pretrained(config.encoder, config.model, source=snapshot.path)
        .to(device)
        .eval()
    )
    validate_context_length(model.encoder, config)
    encoded = tokenizer(
        "Ali Toshkentdagi ACME ofisiga bordi.",
        return_tensors="pt",
        truncation=True,
        max_length=min(32, config.tokenization.max_length),
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    token_types = encoded.get("token_type_ids")
    if token_types is not None:
        token_types = token_types.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_ids, attention_mask, token_types)
    result = {
        "run_id": config.run_id,
        "encoder": config.encoder.name,
        "revision": config.encoder.revision,
        "parameters": _parameter_count(model),
        "tokens": int(input_ids.shape[1]),
        "logits_shape": list(output.logits.shape),
        "dtype": str(output.logits.dtype),
        "device": torch.cuda.get_device_name(device),
        "status": "ok",
    }
    del output, model, tokenizer, encoded
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    """Проверяет все encoder-ы в выбранном этапе по очереди."""
    args = parse_args()
    root = args.project_root.resolve()
    series = load_series((root / args.series).resolve())
    device = resolve_device(True)
    for config_name in series.select(args.stage):
        result = _check_one((root / config_name).resolve(), device)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
