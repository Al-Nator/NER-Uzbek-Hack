"""Проверки честного голосования в пространстве исходных character spans."""

from dataclasses import replace

import pytest

from uzner.domain import Document, Entity, Prediction
from uzner.evaluation.span_vote import SpanVoteConfig, majority_vote


def test_majority_ignores_gold_and_confidence():
    """Два точных голоса побеждают один высокий score; gold не участвует в выборе."""
    doc = Document("a", "Тошкент Ali")
    a, b = Entity(0, 7, "GEO", 0.01), Entity(8, 11, "NAME", 0.99)
    sources = ((Prediction("a", (a,)),), (Prediction("a", (a, b)),), (Prediction("a", (b,)),))
    result = majority_vote((doc,), sources)
    assert [(e.start, e.end, e.label) for e in result[0].entities] == [
        (0, 7, "GEO"),
        (8, 11, "NAME"),
    ]
    assert majority_vote((replace(doc, entities=(a,)),), sources) == result


def test_bad_coverage_and_overlaps():
    """Неполные входы и пересечения нельзя замаскировать голосованием."""
    doc = Document("a", "abc")
    valid = (Prediction("a", ()),)
    with pytest.raises(ValueError):
        majority_vote((doc,), (valid, valid, ()))
    bad = (Prediction("a", (Entity(0, 2, "GEO"), Entity(1, 3, "NAME"))),)
    with pytest.raises(ValueError):
        majority_vote((doc,), (valid, valid, bad))
    with pytest.raises(ValueError):
        SpanVoteConfig(("same", "same", "other"))
