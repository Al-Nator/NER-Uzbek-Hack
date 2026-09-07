"""Последовательный инференс проверенного ансамбля без обучения и gold-метрик."""

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from uzner.config import ExperimentConfig
from uzner.data.io import load_documents, sha256_file, write_predictions
from uzner.data.windows import build_window_features
from uzner.domain import Document, Prediction
from uzner.evaluation.span_vote import SpanVoteConfig, majority_vote
from uzner.experiments.artifacts import write_json
from uzner.training.checkpoint import load_model_checkpoint
from uzner.training.data_setup import make_loader
from uzner.training.inference import run_predictions


@dataclass(frozen=True)
class EnsembleSubmission:
    """Пути фиксированного ансамбля, входного теста и нового каталога результата."""

    ensemble_run: Path
    input_path: Path
    output_dir: Path
    batch_size: int = 8

    def __post_init__(self) -> None:
        """Запрещает некорректный размер батча."""
        if self.batch_size < 1:
            raise ValueError("batch_size должен быть положительным")


def validate_submission(
    documents: tuple[Document, ...], predictions: tuple[Prediction, ...]
) -> None:
    """Проверяет уникальные hash, порядок, плоские сущности и Unicode-границы."""
    hashes = tuple(document.hash for document in documents)
    if len(set(hashes)) != len(hashes):
        raise ValueError("Повторный входной hash")
    if hashes != tuple(prediction.hash for prediction in predictions):
        raise ValueError("Не совпали количество или порядок hash")
    for document, prediction in zip(documents, predictions, strict=True):
        Document(document.hash, document.text, prediction.entities)


def predict_checkpoint(
    source: Path,
    documents: tuple[Document, ...],
    batch_size: int,
    *,
    window_mode: str = "uniform",
) -> tuple[Prediction, ...]:
    """Применяет best-checkpoint с его настройками головы, окон и декодера."""
    checkpoint = source / "checkpoints/best"
    config = ExperimentConfig.from_mapping(
        json.loads((checkpoint / "experiment_config.json").read_text("utf-8"))
    )
    device = torch.device("cuda")
    loaded = load_model_checkpoint(checkpoint, config, device)
    loaded.model.requires_grad_(False)
    predictions = []
    for start in range(0, len(documents), 128):
        group = documents[start : start + 128]
        features = build_window_features(
            group,
            loaded.tokenizer,
            config.tokenization,
            config.model.tag_scheme,
            with_labels=False,
        )
        if window_mode == "word_boundary":
            from uzner.data.inference_windows import word_boundary_features

            features = word_boundary_features(group, loaded.tokenizer, config.tokenization)
        elif window_mode not in {"uniform", "center"}:
            raise ValueError("Неизвестный вариант inference окон")
        loader = make_loader(
            features,
            loaded.tokenizer,
            batch_size=batch_size,
            shuffle=False,
            seed=config.training.seed,
            num_workers=0,
        )
        result = run_predictions(
            loaded.model,
            loader,
            group,
            device,
            bf16=config.training.bf16,
            span_weighting="center" if window_mode == "center" else "uniform",
        )
        predictions.extend(result.predictions)
        print(f"{source.name}: {len(predictions)}/{len(documents)}", flush=True)
    result_predictions = tuple(predictions)
    validate_submission(documents, result_predictions)
    return result_predictions


def checkpoint_hashes(source: Path) -> dict[str, str]:
    """Фиксирует реальные веса и tokenizer, исключая ненужный optimizer state."""
    checkpoint = source / "checkpoints/best"
    return {
        str(path.relative_to(checkpoint)): sha256_file(path)
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file() and path.name != "training_state.pt"
    }


def build_ensemble_submission(request: EnsembleSubmission) -> Path:
    """Сохраняет три компонента, большинство 2/3 и воспроизводимый manifest."""
    if request.output_dir.exists():
        raise FileExistsError(request.output_dir)
    config_path = request.ensemble_run / "resolved_config.json"
    payload = json.loads(config_path.read_text("utf-8"))
    if payload.get("rule") != "exact span 2 of 3":
        raise ValueError("Поддерживается только фиксированное exact-span большинство 2/3")
    vote = SpanVoteConfig(tuple(payload["sources"]))
    sources = tuple(request.ensemble_run.parent / name for name in vote.sources)
    documents = tuple(load_documents((("public_test", request.input_path),)))
    if not documents:
        raise ValueError("Пустой входной файл")
    for source in sources:
        if not (source / "checkpoints/best/head.safetensors").is_file():
            raise FileNotFoundError(source / "checkpoints/best")
    if not torch.cuda.is_available():
        raise RuntimeError("Для полного инференса требуется CUDA GPU")
    request.output_dir.mkdir(parents=True)
    components = []
    provenance = []
    for source in sources:
        hashes = checkpoint_hashes(source)
        component = predict_checkpoint(source, documents, request.batch_size)
        component_path = request.output_dir / "components" / f"{source.name}.jsonl"
        write_predictions(component_path, component)
        components.append(component)
        provenance.append(
            {
                "run_id": source.name,
                "checkpoint": "best",
                "checkpoint_sha256": hashes,
                "predictions_sha256": sha256_file(component_path),
            }
        )
        gc.collect()
        torch.cuda.empty_cache()
    predictions = majority_vote(documents, tuple(components))
    validate_submission(documents, predictions)
    output = request.output_dir / "predictions.jsonl"
    write_predictions(output, predictions)
    write_json(
        output.with_suffix(".manifest.json"),
        {
            "ensemble_run": request.ensemble_run.name,
            "rule": payload["rule"],
            "ensemble_config_sha256": sha256_file(config_path),
            "sources": provenance,
            "input_sha256": sha256_file(request.input_path),
            "output_sha256": sha256_file(output),
            "documents": len(documents),
            "entities": sum(len(p.entities) for p in predictions),
            "batch_size": request.batch_size,
            "device": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "training": False,
            "gold_evaluation": False,
        },
    )
    return output
