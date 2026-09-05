"""Общее объединение span-кандидатов и плоский exact-offset decoder."""

from bisect import bisect_left
from dataclasses import dataclass

import torch

from uzner.data.spans import boundary_indices, span_pair_mask
from uzner.data.windows import Offset
from uzner.domain import LABELS, Entity, Prediction


@dataclass(frozen=True, slots=True)
class SpanWindowScores:
    """Вероятности одного окна в исходных character offsets."""

    offsets: tuple[Offset, ...]
    probabilities: torch.Tensor


def select_flat_entities(entities: list[Entity]) -> tuple[Entity, ...]:
    """Жадно выбирает непересекающиеся spans по score с устойчивыми tie-breaks."""
    selected, starts = [], []
    for entity in sorted(
        entities,
        key=lambda item: (-(item.score or 0), -(item.end - item.start), item.start, item.label),
    ):
        index = bisect_left(starts, entity.start)
        if index and selected[index - 1].end > entity.start:
            continue
        if index < len(selected) and selected[index].start < entity.end:
            continue
        selected.insert(index, entity)
        starts.insert(index, entity.start)
    return tuple(selected)


def decode_span_windows(
    document_hash: str, windows: list[SpanWindowScores], threshold: float
) -> Prediction:
    """Усредняет все покрывающие окно оценки, включая отрицательные голоса."""
    candidates = set()
    indexed = []
    for window in windows:
        starts, ends = boundary_indices(window.offsets)
        width = len(window.offsets)
        valid = span_pair_mask(window.offsets, width)
        probabilities = window.probabilities[:, :width, :width]
        indexed.append((starts, ends, probabilities, valid))
        for label, start, end in torch.nonzero(
            (probabilities > threshold) & valid[None], as_tuple=False
        ).tolist():
            candidates.add((label, window.offsets[start][0], window.offsets[end][1]))
    entities = []
    for label, start, end in sorted(candidates):
        votes = []
        for starts, ends, probabilities, valid in indexed:
            left, right = starts.get(start), ends.get(end)
            if left is not None and right is not None and valid[left, right]:
                votes.append(float(probabilities[label, left, right]))
        score = sum(votes) / len(votes)
        if score > threshold:
            entities.append(Entity(start, end, LABELS[label], score))
    return Prediction(document_hash, select_flat_entities(entities))


@dataclass(slots=True)
class SpanPredictionAccumulator:
    """Хранит матрицы только текущего документа при упорядоченном dev-loader."""

    threshold: float
    hashes: tuple[str, ...]
    predictions: list[Prediction]
    windows: list[SpanWindowScores]
    document_index: int = 0

    def add(self, index: int, window: SpanWindowScores) -> None:
        """Завершает предыдущий документ и принимает следующее окно."""
        if index < self.document_index:
            raise ValueError("Span inference требует последовательного порядка документов")
        while self.document_index < index:
            self._flush()
        self.windows.append(window)

    def _flush(self) -> None:
        """Декодирует один документ и освобождает его квадратичные матрицы."""
        self.predictions.append(
            decode_span_windows(self.hashes[self.document_index], self.windows, self.threshold)
        )
        self.windows.clear()
        self.document_index += 1

    def finish(self) -> tuple[Prediction, ...]:
        """Завершает последние и пустые документы в исходном порядке."""
        while self.document_index < len(self.hashes):
            self._flush()
        return tuple(self.predictions)
