"""Проверки кэша до NMS и неизменности штатного GP-декодера."""

import torch

from uzner.training.span_prediction import (
    SpanWindowScores,
    aggregate_span_candidates,
    decode_span_windows,
    select_flat_entities,
)


def test_pre_nms_preserves_overlap_and_negative_votes():
    """Кэш содержит пересечения, но учитывает отрицательное покрытие окон."""
    offsets = ((0, 1), (2, 3), (4, 5))
    scores = torch.zeros(3, 3, 3)
    scores[0, 0, 0], scores[0, 0, 1], scores[1, 2, 2] = 0.9, 0.8, 0.7
    windows = [SpanWindowScores(offsets, scores), SpanWindowScores(offsets, scores * 0.5)]
    candidates = aggregate_span_candidates(windows, 0.05)
    assert len(candidates) == 3
    assert abs(candidates[0].score - 0.675) < 1e-6
    for threshold in (0.05, 0.2, 0.5, 0.6, 0.8):
        cached = select_flat_entities([e for e in candidates if e.score > threshold])
        assert cached == decode_span_windows("doc", windows, threshold).entities


def test_empty_candidates():
    """Пустой документ не создаёт фиктивных сущностей."""
    assert aggregate_span_candidates([], 0.05) == ()
