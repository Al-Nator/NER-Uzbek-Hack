"""MELM-inspired подготовка train-копий с frozen-NER фильтром и MLflow-аудитом."""

import gc
import json
from dataclasses import asdict
from pathlib import Path

import mlflow
import torch

from uzner.data.io import load_documents, sha256_file
from uzner.evaluation.slices import normalize_surface
from uzner.experiments.artifacts import write_json, write_jsonl
from uzner.training.frozen import FrozenPredictor
from uzner.training.melm_generator import MelmConfig, generate, train_generator
from uzner.training.runtime import set_reproducible_seed


def prepare_melm(output: Path, source: Path, config: MelmConfig, *, smoke: bool = False) -> Path:
    """Создаёт новый релиз; dev используется только для исключения совпадений текста."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    train_path = Path("ner_uz_hackathon_participant/data/train.jsonl")
    dev_path = Path("ner_uz_hackathon_participant/data/dev.jsonl")
    train = tuple(load_documents((("official", train_path),)))
    if smoke:
        train = train[:16]
    # Не передаём метки dev ни генератору, ни фильтру.
    dev_texts = {
        normalize_surface(json.loads(s)["text"]) for s in dev_path.read_text().splitlines()
    }
    blocked = dev_texts | {normalize_surface(d.text) for d in train}
    set_reproducible_seed(config.seed)
    manifest = {
        "config": asdict(config),
        "method": "MELM-inspired-single-entity-context",
        "train_sha256": sha256_file(train_path),
        "dev_sha256": sha256_file(dev_path),
        "filter_source": str(source),
        "filter_head_sha256": sha256_file(source / "checkpoints/best/head.safetensors"),
        "selection_uses_dev_labels": False,
        "language_quality_verified": False,
    }
    write_json(output / "manifest.json", manifest)
    mlflow.set_tracking_uri("sqlite:///" + str(Path("mlruns/mlflow.db").resolve()))
    if not smoke:
        mlflow.set_experiment("uzner-data-preparation")
        mlflow.start_run(
            run_name=output.name,
            tags={"uzner.kind": "data_generation", "uzner.method": "MELM-inspired"},
        )
        mlflow.log_params(asdict(config))
        mlflow.log_params({"train_sha256": manifest["train_sha256"], "filter_source": str(source)})
    tracker = SilentTracker() if smoke else mlflow
    try:
        model, tokenizer, examples = train_generator(train, output, config, tracker)
        candidates = generate(model, tokenizer, examples, train, config)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        candidates = [c for c in candidates if normalize_surface(c[0].text) not in blocked]
        predictor = FrozenPredictor(source)
        accepted = []
        seen = set(blocked)
        for offset in range(0, len(candidates), 128):
            batch = candidates[offset : offset + 128]
            predictions = predictor.predict(tuple(c[0] for c in batch))
            for (doc, source_hash, index), prediction in zip(batch, predictions, strict=True):
                expected = doc.entities[index]
                matches = any(
                    (e.label, e.start, e.end) == (expected.label, expected.start, expected.end)
                    for e in prediction.entities
                )
                key = normalize_surface(doc.text)
                if matches and key not in seen:
                    accepted.append(
                        {
                            "hash": doc.hash,
                            "text": doc.text,
                            "entities": [e.to_mapping() for e in doc.entities],
                            "source_hash": source_hash,
                            "method": "melm_inspired",
                            "entity_index": index,
                            "language_approved": False,
                            "filter": "edited_entity_exact_agreement",
                        }
                    )
                    seen.add(key)
            print(
                f"MELM filter {min(offset + 128, len(candidates))}/{len(candidates)} "
                f"accepted={len(accepted)}",
                flush=True,
            )
            if len(accepted) >= config.max_accepted:
                break
        accepted = accepted[: config.max_accepted]
        if not smoke and len(accepted) < 500:
            raise RuntimeError(
                f"Слишком мало прошедших фильтр кандидатов: {len(accepted)}; NER не запускается"
            )
        path = output / "train_augmented.jsonl"
        write_jsonl(path, accepted)
        manifest.update(
            {
                "generated": len(candidates),
                "accepted": len(accepted),
                "sha256": sha256_file(path),
                "status": "complete",
            }
        )
        write_json(output / "manifest.json", manifest)
        if not smoke:
            mlflow.log_metrics({"data/generated": len(candidates), "data/accepted": len(accepted)})
            mlflow.log_artifact(str(output / "manifest.json"))
            mlflow.end_run()
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
        return path
    except Exception:
        if not smoke:
            mlflow.end_run(status="FAILED")
        raise


class SilentTracker:
    """Не публикует smoke-проверки как эксперименты."""

    def log_params(self, values: dict) -> None:
        """Принимает параметры без внешней записи."""

    def log_metrics(self, values: dict, *, step: int | None = None) -> None:
        """Принимает метрики без внешней записи."""
