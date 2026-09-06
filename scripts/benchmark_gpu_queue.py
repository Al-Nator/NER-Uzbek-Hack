"""Короткая проверка throughput реального GP без checkpoint-ов и MLflow."""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from uzner.config import load_experiment_config
from uzner.data.spans import with_span_targets
from uzner.data.windows import build_window_features
from uzner.models.factory import model_class
from uzner.models.pretrained import resolve_pretrained_snapshot
from uzner.training.data_setup import load_split, make_loader
from uzner.training.optimizer import make_optimizer
from uzner.training.runtime import set_reproducible_seed


def main() -> None:
    """Измеряет прогретые шаги на official train с настоящими целями и AdamW."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    if args.steps < 25:
        raise ValueError("Нужно не меньше 25 измеряемых шагов")
    config = load_experiment_config(args.config)
    set_reproducible_seed(config.training.seed)
    torch.set_num_threads(4)
    source = resolve_pretrained_snapshot(config.encoder).path
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, fix_mistral_regex=False
    )
    documents = load_split(config, Path.cwd(), "train", None)
    features = with_span_targets(
        build_window_features(documents, tokenizer, config.tokenization, "bio", with_labels=False),
        documents,
    )
    loader = make_loader(features, tokenizer, batch_size=8, shuffle=True, seed=42, num_workers=4)
    model = model_class(config.model).from_pretrained(config.encoder, config.model, source=source)
    model = model.cuda().train()
    optimizer = make_optimizer(model, config.training)
    tokens = 0
    started = None
    for index, batch in enumerate(loader):
        if index == args.steps + 10:
            break
        batch = batch.to(torch.device("cuda"))
        if index == 10:
            torch.cuda.synchronize()
            started = time.time()
            print(json.dumps({"phase": "measured_start", "time": started}), flush=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                batch.input_ids, batch.attention_mask, batch.token_type_ids, batch.labels
            )
        if not torch.isfinite(output.loss):
            raise RuntimeError("Неконечный loss в benchmark")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if index >= 10:
            tokens += int(batch.attention_mask.sum())
    torch.cuda.synchronize()
    elapsed = time.time() - started
    print(
        json.dumps(
            {
                "steps": args.steps,
                "seconds": elapsed,
                "tokens": tokens,
                "tokens_per_second": tokens / elapsed,
                "finished": time.time(),
                "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
