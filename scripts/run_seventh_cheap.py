"""Отдельные дешёвые абляции замены голоса, веса окон и границ нарезки."""

import gc
from pathlib import Path

import torch
import yaml

from uzner.data.io import load_documents, write_predictions
from uzner.domain import Document
from uzner.evaluation.span_vote import majority_vote
from uzner.experiments.research_run import ResearchRun
from uzner.training.reranker_cache import CANONICAL, read_predictions
from uzner.training.submission import predict_checkpoint


def main() -> None:
    """Применяет каждую абляцию независимо, не заменяя канонический s62."""
    path = Path("configs/seventh/cheap.yaml")
    config = yaml.safe_load(path.read_text())
    official = Path("ner_uz_hackathon_participant/data")
    dev = tuple(load_documents((("dev", official / "dev.jsonl"),)))
    train = tuple(load_documents((("train", official / "train.jsonl"),)))
    unlabeled = tuple(Document(d.hash, d.text) for d in dev)
    for item in config["variants"]:
        root = Path("runs") / item["run_id"]
        files = tuple(Path("runs") / name / "predictions/dev.jsonl" for name in item["sources"])
        with ResearchRun(
            root,
            item,
            (path, *files, official / "dev.jsonl", official / "train.jsonl"),
            "inference_ablation",
        ) as run:
            sources = []
            for index, name in enumerate(item["sources"]):
                if item["mode"] == "cached" or index == 1:
                    result = read_predictions(files[index])
                else:
                    result = predict_checkpoint(
                        Path("runs") / name, unlabeled, 8, window_mode=item["mode"]
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                sources.append(result)
                write_predictions(root / "components" / f"{name}.jsonl", result)
            predictions = majority_vote(dev, tuple(sources))
            run.evaluate(dev, predictions, train, "dev")
            reference = tuple(
                read_predictions(Path("runs") / name / "predictions/dev.jsonl")
                for name in CANONICAL
            )
            run.evaluate(dev, majority_vote(dev, reference), train, "reference_s62")


if __name__ == "__main__":
    main()
