"""Предсказания новых holdout-proposer-ов без доступа к meta gold при инференсе."""

import gc
import json
from pathlib import Path

import torch

from uzner.config import ExperimentConfig
from uzner.data.io import load_documents, read_jsonl, sha256_file, write_predictions
from uzner.domain import Document, Prediction
from uzner.experiments.artifacts import write_json
from uzner.training.data_setup import load_split
from uzner.training.submission import predict_checkpoint

PROPOSERS = ("s7p0_bge_holdout_v1", "s7p1_mdeberta_holdout_v1", "s7p2_xlmr_holdout_v1")
CANONICAL = (
    "s33_bge_global_pointer_low_lr_a100-low-lr-v1",
    "s21_mdeberta_v3_base_bioes_crf_s2-sequence-v1",
    "s31_xlmr_large_global_pointer_a100-continuation-v1",
)


def read_predictions(path: Path) -> tuple[Prediction, ...]:
    """Читает общий формат предсказаний без собственных правил декодирования."""
    return tuple(Prediction.from_mapping(row) for row in read_jsonl(path))


def verify_holdout_source(source: Path, split_root: Path) -> None:
    """Проверяет реально записанные train/validation inputs против split-манифеста."""
    split = json.loads((split_root / "manifest.json").read_text())
    for name, digest in split["output_sha256"].items():
        if sha256_file(split_root / f"{name}.jsonl") != digest:
            raise ValueError(f"Изменён split {name}")
    config = ExperimentConfig.from_mapping(
        json.loads((source / "checkpoints/best/experiment_config.json").read_text())
    )
    if config.training.initial_checkpoint:
        raise ValueError("Полнообученный NER checkpoint запрещён для holdout proposer")
    metadata = json.loads((source / "metadata.json").read_text())
    for name, key in (("proposer_train", "train"), ("proposer_valid", "dev")):
        actual = load_split(config, Path.cwd(), key, None)
        expected = {d.hash for d in load_documents(((name, split_root / f"{name}.jsonl"),))}
        if {d.hash for d in actual} != expected:
            raise ValueError("Proposer обучался на другом наборе документов")
        if metadata["data"][name]["sha256"] != split["output_sha256"][name]:
            raise ValueError("Данные proposer изменены после обучения")


def prepare_prediction_cache(split_root: Path) -> Path:
    """Создаёт полные meta/внешние контрольные прогнозы трёх новых моделей."""
    cache = split_root / "cache"
    manifest_path = cache / "manifest.json"
    for name in PROPOSERS:
        verify_holdout_source(Path("runs") / name, split_root)
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        for path, digest in saved["files"].items():
            if sha256_file(Path(path)) != digest:
                raise ValueError("Изменён cache reranker")
        return cache
    if cache.exists():
        raise FileExistsError("Незавершённый cache: требуется явная проверка перед повтором")
    cache.mkdir()
    split_documents = {
        name: tuple(load_documents(((name, split_root / f"{name}.jsonl"),)))
        for name in ("meta_train", "meta_valid")
    }
    split_documents["heldout_dev"] = tuple(
        load_documents((("dev", Path("ner_uz_hackathon_participant/data/dev.jsonl")),))
    )
    inputs = [split_root / "manifest.json"]
    for source_name in PROPOSERS:
        source = Path("runs") / source_name
        verify_holdout_source(source, split_root)
        inputs.append(source / "artifact_manifest.json")
        inputs.extend((source / "checkpoints/best").rglob("*.safetensors"))
        for split, documents in split_documents.items():
            unlabeled = tuple(Document(d.hash, d.text) for d in documents)
            result = predict_checkpoint(source, unlabeled, 8)
            write_predictions(cache / split / f"{source_name}.jsonl", result)
            gc.collect()
            torch.cuda.empty_cache()
    for name in CANONICAL:
        path = split_root / "canonical_dev" / f"{name}.jsonl"
        inputs.append(path)
    files = [*inputs, *cache.rglob("*.jsonl")]
    write_json(
        manifest_path,
        {
            "files": {str(p): sha256_file(p) for p in files},
            "protocol": "train-holdout pilot; not full OOF",
        },
    )
    return cache
