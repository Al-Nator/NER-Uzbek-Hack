"""Запуск трёх вариантов span-reranker после готовности holdout-proposer-ов."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from uzner.data.io import load_documents
from uzner.experiments.research_run import ResearchRun
from uzner.models.span_reranker import RerankerConfig
from uzner.training.reranker_cache import (
    CANONICAL,
    PROPOSERS,
    prepare_prediction_cache,
    read_predictions,
)
from uzner.training.reranker_fit import fit_reranker, make_reranker_dataset


def main() -> None:
    """Проверяет provenance cache и выполняет только ещё не завершённые варианты."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/seventh/rerankers.yaml"))
    parser.add_argument("--wait-proposers", action="store_true")
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    split_root = Path(raw["split_root"])
    deadline = time.monotonic() + 6 * 3600
    while args.wait_proposers:
        if time.monotonic() > deadline:
            raise TimeoutError("Proposer-ы не готовы за 6 часов; проверьте очередь и место")
        states = [
            json.loads((Path("runs") / name / "status.json").read_text())
            if (Path("runs") / name / "status.json").exists()
            else {}
            for name in PROPOSERS
        ]
        if any(s.get("status") == "failed" for s in states):
            raise RuntimeError("Proposer завершился ошибкой; reranker не запускается")
        if all(
            s.get("status") == "complete" and s.get("kind") == "reranker_proposer" for s in states
        ):
            break
        print("Ожидаю завершения трёх holdout-proposer-ов", flush=True)
        time.sleep(60)
    if not torch.cuda.is_available():
        raise RuntimeError("Полный reranker эксперимент требует GPU")
    cache = prepare_prediction_cache(split_root)
    datasets = {}
    for split in ("meta_train", "meta_valid", "heldout_dev", "dev"):
        path = (
            split_root / f"{split}.jsonl"
            if split.startswith("meta_")
            else Path("ner_uz_hackathon_participant/data/dev.jsonl")
        )
        documents = tuple(load_documents(((split, path),)))
        paths = (
            [cache / split / f"{name}.jsonl" for name in PROPOSERS]
            if split != "dev"
            else [split_root / "canonical_dev" / f"{name}.jsonl" for name in CANONICAL]
        )
        datasets[split] = make_reranker_dataset(
            documents, tuple(read_predictions(p) for p in paths)
        )
    reference = tuple(
        load_documents((("train", Path("ner_uz_hackathon_participant/data/train.jsonl")),))
    )
    for variant in raw["variants"]:
        config = RerankerConfig(**variant)
        root = Path("runs") / config.run_id
        if root.exists():
            status = json.loads((root / "status.json").read_text())
            if status.get("status") == "complete":
                continue
            raise FileExistsError(root)
        with ResearchRun(
            root,
            {
                **asdict(config),
                "selection_split": "meta_valid",
                "protocol": "train-holdout pilot, not full OOF",
                "dev_proposers": list(CANONICAL),
                "holdout_proposers": list(PROPOSERS),
            },
            (args.config, cache / "manifest.json", split_root / "manifest.json"),
            "span_reranker",
        ) as run:
            fit_reranker(
                config,
                datasets["meta_train"],
                datasets["meta_valid"],
                {name: datasets[name] for name in ("heldout_dev", "dev")},
                reference,
                run,
                torch.device("cuda"),
            )


if __name__ == "__main__":
    main()
