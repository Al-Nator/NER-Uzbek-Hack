"""Полная проверка token alignment выбранной серии до обучения."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from transformers import AutoTokenizer

from uzner.config import ExperimentConfig, load_data_config, load_experiment_config, resolve_sources
from uzner.data.io import load_documents
from uzner.data.spans import span_coverage, with_span_targets
from uzner.data.windows import IGNORE_LABEL, build_window_features
from uzner.experiments.series import load_series
from uzner.models.pretrained import resolve_pretrained_snapshot


@dataclass(frozen=True, slots=True)
class AlignmentSummary:
    """Итог построения окон одного split-а."""

    run_id: str
    split: str
    documents: int
    windows: int
    supervised_tokens: int
    masked_tokens: int
    gold_spans: int | None = None
    represented_spans: int | None = None


def parse_args() -> argparse.Namespace:
    """Разбирает манифест, этап и корень проекта."""
    parser = argparse.ArgumentParser(description="Проверить alignment на всех документах")
    parser.add_argument("--series", type=Path, default=Path("configs/series/first.yaml"))
    parser.add_argument("--stage", default="all")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def _signature(config: ExperimentConfig) -> tuple[object, ...]:
    """Строит ключ уникального tokenizer/alignment-варианта."""
    return (
        config.encoder.name,
        config.encoder.revision,
        config.encoder.use_safetensors,
        config.tokenization.max_length,
        config.tokenization.stride,
        config.model.tag_scheme,
        config.model.architecture,
        config.data_config,
    )


def _check_split(
    config: ExperimentConfig,
    tokenizer: object,
    project_root: Path,
    split: str,
) -> AlignmentSummary:
    """Строит все окна split-а и считает supervised/masked токены."""
    data = load_data_config((project_root / config.data_config).resolve())
    documents = tuple(load_documents(resolve_sources(data, split=split, project_root=project_root)))
    features = build_window_features(
        documents,
        tokenizer,
        config.tokenization,
        config.model.tag_scheme,
        with_labels=config.model.architecture == "token_tagging",
    )
    if config.model.architecture == "span":
        span_features = with_span_targets(features, documents)
        coverage = span_coverage(span_features, documents)
        return AlignmentSummary(
            config.run_id,
            split,
            len(documents),
            len(features),
            sum(
                sum(start < end for start, end in item.offsets) - len(item.ignored_tokens)
                for item in span_features
            ),
            sum(len(item.ignored_tokens) for item in span_features),
            coverage["gold"],
            coverage["represented"],
        )
    labels = (label for feature in features for label in feature.labels or ())
    supervised = 0
    masked = 0
    for label in labels:
        if label == IGNORE_LABEL:
            masked += 1
        else:
            supervised += 1
    return AlignmentSummary(
        run_id=config.run_id,
        split=split,
        documents=len(documents),
        windows=len(features),
        supervised_tokens=supervised,
        masked_tokens=masked,
    )


def main() -> int:
    """Проверяет каждый уникальный вариант и печатает JSONL-итоги."""
    args = parse_args()
    root = args.project_root.resolve()
    series = load_series((root / args.series).resolve())
    checked: dict[tuple[object, ...], str] = {}
    for config_name in series.select(args.stage):
        config = load_experiment_config((root / config_name).resolve())
        if config.pipeline != "uzner":
            print(json.dumps({"run_id": config.run_id, "status": "official-skip"}))
            continue
        signature = _signature(config)
        if signature in checked:
            print(
                json.dumps(
                    {"run_id": config.run_id, "status": "duplicate", "same_as": checked[signature]}
                )
            )
            continue
        snapshot = resolve_pretrained_snapshot(config.encoder)
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot.path,
            trust_remote_code=config.encoder.trust_remote_code,
            use_fast=True,
            local_files_only=True,
            fix_mistral_regex=False,
        )
        for split in ("train", "dev"):
            summary = _check_split(config, tokenizer, root, split)
            print(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True))
        checked[signature] = config.run_id
        del tokenizer
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
