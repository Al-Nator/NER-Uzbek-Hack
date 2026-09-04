"""CLI аудита границ и фрагментации tokenizer-а до обучения."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from uzner.config import load_data_config, load_experiment_config, resolve_sources
from uzner.data.io import load_documents
from uzner.evaluation.tokenizer_audit import audit_tokenizer
from uzner.experiments.artifacts import write_json
from uzner.models.pretrained import resolve_pretrained_snapshot


def parse_args() -> argparse.Namespace:
    """Разбирает experiment, split и необязательный output."""
    parser = argparse.ArgumentParser(description="Проверить tokenizer на exact spans")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Строит аудит и печатает или сохраняет JSON."""
    args = parse_args()
    root = args.project_root.resolve()
    config = load_experiment_config(args.config.resolve())
    data = load_data_config((root / config.data_config).resolve())
    documents = tuple(load_documents(resolve_sources(data, split=args.split, project_root=root)))
    snapshot = resolve_pretrained_snapshot(config.encoder)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot.path,
        trust_remote_code=config.encoder.trust_remote_code,
        use_fast=True,
        local_files_only=True,
        fix_mistral_regex=False,
    )
    result = audit_tokenizer(documents, tokenizer, config.encoder, config.tokenization)
    payload = result.to_mapping()
    if args.output is None:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        write_json(args.output.resolve(), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
