"""Полный dev-инференс с объединением окон и slice-метриками."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from uzner.data.windows import WindowBatch, WindowFeature, collect_chunk_edges
from uzner.domain import Document, Prediction
from uzner.evaluation.slices import DetailedEvaluation, evaluate_detailed
from uzner.models.token_tagger import TokenTagger
from uzner.training.prediction import (
    WindowLogits,
    aggregate_window_logits,
    decode_documents,
)
from uzner.training.span_prediction import SpanPredictionAccumulator, SpanWindowScores


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Предсказания, метрики и скорость одного dev-прогона."""

    predictions: tuple[Prediction, ...]
    evaluation: DetailedEvaluation
    loss: float
    seconds: float
    documents_per_second: float


def _autocast(device: torch.device, bf16: bool) -> torch.autocast:
    """Создаёт BF16 autocast только для CUDA-вычислений."""
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=bf16 and device.type == "cuda",
    )


@torch.inference_mode()
def run_inference(
    model: TokenTagger,
    loader: DataLoader[WindowBatch],
    features: tuple[WindowFeature, ...],
    documents: tuple[Document, ...],
    train_documents: tuple[Document, ...],
    device: torch.device,
    *,
    bf16: bool,
) -> InferenceResult:
    """Считает loss, exact spans и все диагностические метрики."""
    model.eval()
    started = perf_counter()
    window_outputs: list[WindowLogits] = []
    span_output = (
        SpanPredictionAccumulator(
            model.model_config.span_threshold,
            tuple(document.hash for document in documents),
            [],
            [],
        )
        if model.model_config.architecture == "span"
        else None
    )
    weighted_loss = 0.0
    loss_weight = 0
    for batch in loader:
        batch = batch.to(device)
        with _autocast(device, bf16):
            output = model(
                batch.input_ids,
                batch.attention_mask,
                batch.token_type_ids,
                batch.labels,
            )
        if output.loss is not None:
            weight = len(batch.document_indices)
            weighted_loss += float(output.loss.item()) * weight
            loss_weight += weight
        logits = output.logits.detach().float().cpu()
        if span_output is not None:
            probabilities = (
                logits.softmax(dim=1)[:, 1:]
                if model.model_config.head == "biaffine"
                else logits.sigmoid()
            )
            for row, index, offsets in zip(
                probabilities, batch.document_indices, batch.offsets, strict=True
            ):
                span_output.add(
                    index, SpanWindowScores(offsets, row[:, : len(offsets), : len(offsets)].clone())
                )
            continue
        for row, document_index, offsets in zip(
            logits, batch.document_indices, batch.offsets, strict=True
        ):
            window_outputs.append(
                WindowLogits(
                    document_index=document_index,
                    offsets=offsets,
                    logits=row[: len(offsets)],
                )
            )
    if span_output is not None:
        predictions = span_output.finish()
    else:
        emissions = aggregate_window_logits(
            tuple(window_outputs),
            len(documents),
            average_probabilities=model.model_config.head == "softmax",
        )
        predictions = decode_documents(documents, emissions, model)
    raw_edges = collect_chunk_edges(features)
    edges = {document.hash: raw_edges.get(index, ()) for index, document in enumerate(documents)}
    evaluation = evaluate_detailed(documents, predictions, train_documents, edges)
    seconds = perf_counter() - started
    return InferenceResult(
        predictions=predictions,
        evaluation=evaluation,
        loss=weighted_loss / loss_weight if loss_weight else 0.0,
        seconds=seconds,
        documents_per_second=len(documents) / seconds if seconds else 0.0,
    )
