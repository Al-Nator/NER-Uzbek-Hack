"""Train-only символьные цели и gold-free признаки кандидатов."""

import random
from dataclasses import dataclass

import torch

from uzner.domain import LABELS, Document, Entity, Prediction
from uzner.models.char_boundary import CharConfig, character_window
from uzner.training.frozen import FrozenPredictor
from uzner.training.span_prediction import select_flat_entities


@dataclass(frozen=True)
class BoundaryExample:
    """Одна граница с контекстом encoder-а и необязательной train-целью."""

    document: int
    entity: int
    side: int
    anchor: int
    hidden: torch.Tensor
    chars: tuple[int, ...]
    valid: tuple[bool, ...]
    kind: int
    target: int = -100


def collect_boundaries(
    predictor: FrozenPredictor,
    documents: tuple[Document, ...],
    alphabet: dict[str, int],
    config: CharConfig,
    *,
    training: bool,
    predictions: tuple[Prediction, ...] | None = None,
) -> list[BoundaryExample]:
    """Создаёт train-цели из gold; inference читает только модельные кандидаты."""
    if not training and predictions is None:
        raise ValueError("Inference требует модельных кандидатов")
    rng = random.Random(config.seed)
    examples, seen = [], set()
    for window in predictor.windows(documents):
        index = window.document_index
        doc = documents[index]
        entities = doc.entities if training else predictions[index].entities
        tokens = [(t, a, b) for t, (a, b) in enumerate(window.feature.offsets) if b > a]
        if not tokens:
            continue
        lo, hi = min(t[1] for t in tokens), max(t[2] for t in tokens)
        for entity_index, entity in enumerate(entities):
            if entity.start < lo or entity.end > hi:
                continue
            for side, boundary in enumerate((entity.start, entity.end)):
                key = (doc.hash, entity_index, side)
                if key in seen:
                    continue
                token = min(tokens, key=lambda t: abs(t[side + 1] - boundary))
                coarse = token[side + 1]
                if abs(coarse - boundary) > config.radius:
                    continue
                seen.add(key)
                # Inference не знает gold: anchor совпадает с предсказанной границей.
                anchor = coarse if training else boundary
                if training:
                    anchor = max(0, min(len(doc.text), coarse + rng.randint(-2, 2)))
                    if abs(anchor - boundary) > config.radius:
                        anchor = coarse
                chars, valid = character_window(doc.text, anchor, alphabet, config)
                examples.append(
                    BoundaryExample(
                        index,
                        entity_index,
                        side,
                        anchor,
                        window.hidden[token[0]].half().cpu().clone(),
                        chars,
                        valid,
                        2 * LABELS.index(entity.label) + side,
                        boundary - anchor + config.radius if training else -100,
                    )
                )
    if training and len(examples) > config.max_examples:
        examples = rng.sample(examples, config.max_examples)
    return examples


def boundary_batch(examples: list[BoundaryExample], device: torch.device) -> tuple:
    """Собирает компактный batch без encoder-вычислений."""
    return (
        torch.stack([e.hidden for e in examples]).to(device),
        torch.tensor([e.chars for e in examples], device=device),
        torch.tensor([e.kind for e in examples], device=device),
        torch.tensor([e.valid for e in examples], device=device),
        torch.tensor([e.target for e in examples], device=device),
    )


def apply_boundaries(
    predictions: tuple[Prediction, ...],
    examples: list[BoundaryExample],
    positions: list[int],
    confidences: list[float],
    documents: tuple[Document, ...],
    config: CharConfig,
) -> tuple[Prediction, ...]:
    """Применяет уверенные символьные границы, затем общий flat-span decoder."""
    edits = {}
    for example, position, confidence in zip(examples, positions, confidences, strict=True):
        if confidence >= config.confidence:
            edits[(example.document, example.entity, example.side)] = (
                example.anchor + position - config.radius
            )
    output = []
    for index, prediction in enumerate(predictions):
        entities = []
        for i, entity in enumerate(prediction.entities):
            start = edits.get((index, i, 0), entity.start)
            end = edits.get((index, i, 1), entity.end)
            if not 0 <= start < end <= len(documents[index].text):
                start, end = entity.start, entity.end
            entities.append(Entity(start, end, entity.label, entity.score))
        output.append(Prediction(prediction.hash, select_flat_entities(entities)))
    return tuple(output)
