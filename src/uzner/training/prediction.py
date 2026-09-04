"""Объединение перекрывающихся окон и exact-span decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uzner.data.tagging import decode_tags
from uzner.domain import Document, Prediction
from uzner.models.token_tagger import TokenTagger

Offset = tuple[int, int]


@dataclass(frozen=True, slots=True)
class WindowLogits:
    """Недополненные emissions одного окна на CPU."""

    document_index: int
    offsets: tuple[Offset, ...]
    logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class DocumentEmissions:
    """Агрегированные emissions в координатах одного документа."""

    offsets: tuple[Offset, ...]
    values: torch.Tensor


def aggregate_window_logits(
    windows: tuple[WindowLogits, ...],
    document_count: int,
    *,
    average_probabilities: bool,
) -> tuple[DocumentEmissions, ...]:
    """Усредняет повторные токены из соседних окон."""
    if document_count < 1:
        raise ValueError("document_count должен быть положительным")
    aggregated: list[dict[Offset, tuple[torch.Tensor, int]]] = [{} for _ in range(document_count)]
    for window in windows:
        if not 0 <= window.document_index < document_count:
            raise ValueError("Индекс документа окна выходит за границы")
        if window.logits.ndim != 2 or window.logits.shape[0] != len(window.offsets):
            raise ValueError("Количество offsets и emissions окна должно совпадать")
        values = (
            torch.softmax(window.logits.float(), dim=-1)
            if average_probabilities
            else window.logits.float()
        )
        for offset, value in zip(window.offsets, values, strict=True):
            if offset[0] == offset[1]:
                continue
            previous = aggregated[window.document_index].get(offset)
            if previous is None:
                aggregated[window.document_index][offset] = (value.clone(), 1)
            else:
                aggregated[window.document_index][offset] = (previous[0] + value, previous[1] + 1)

    result: list[DocumentEmissions] = []
    for document_scores in aggregated:
        if not document_scores:
            raise ValueError("Tokenizer не вернул токенов для документа")
        ordered = sorted(document_scores.items())
        offsets = tuple(offset for offset, _ in ordered)
        rows = []
        for _, (score_sum, count) in ordered:
            average = score_sum / count
            if average_probabilities:
                average = average.clamp_min(1e-12).log()
            rows.append(average)
        result.append(DocumentEmissions(offsets=offsets, values=torch.stack(rows)))
    return tuple(result)


def decode_documents(
    documents: tuple[Document, ...],
    emissions: tuple[DocumentEmissions, ...],
    model: TokenTagger,
) -> tuple[Prediction, ...]:
    """Преобразует агрегированные emissions в финальные spans."""
    if len(documents) != len(emissions):
        raise ValueError("Количество документов и emissions должно совпадать")
    predictions: list[Prediction] = []
    for document, document_emissions in zip(documents, emissions, strict=True):
        path = model.decode(document_emissions.values)
        tags = tuple(model.tags[index] for index in path)
        entities = decode_tags(
            document_emissions.offsets,
            tags,
            model.model_config.tag_scheme,
        )
        predictions.append(Prediction(hash=document.hash, entities=entities))
    return tuple(predictions)
