"""Контракт точных повторов, Unicode-координат и сохранения исходных ответов."""

import pytest

from uzner.domain import Document, Entity, Prediction
from uzner.evaluation.repeated_mentions import RepeatConfig, propagate_repeats


def test_exact_case_boundaries_and_no_duplicates():
    """Повторы не проникают внутрь слов, суффиксов, дефисов и чужих spans."""
    text = "Artel Artel artel Artelga Artel-Pro Artel's"
    doc = Document("x", text)
    pred = Prediction("x", (Entity(0, 5, "ORG"),))
    result = propagate_repeats(doc, pred)
    assert [(e.start, e.end) for e in result.entities] == [(0, 5), (6, 11)]
    assert propagate_repeats(doc, result) == result


def test_unicode_and_existing_overlap():
    """Координаты считаются в Unicode, существующая сущность не заменяется."""
    doc = Document("x", "Навоий Навоий Навоий")
    pred = Prediction("x", (Entity(0, 6, "NAME"), Entity(7, 13, "GEO")))
    result = propagate_repeats(doc, pred)
    assert result.entities == (*pred.entities, Entity(14, 20, "NAME"))


def test_short_and_invalid_inputs():
    """Короткие строки пропускаются, неверные параметры отклоняются."""
    doc = Document("x", "Ali Ali")
    pred = Prediction("x", (Entity(0, 3, "NAME"),))
    assert propagate_repeats(doc, pred) == pred
    with pytest.raises(ValueError):
        propagate_repeats(doc, Prediction("y", ()))
    with pytest.raises(ValueError):
        propagate_repeats(doc, pred, RepeatConfig(min_length=0))
