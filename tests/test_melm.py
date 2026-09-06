"""Проверки MELM без внешних моделей и публикации smoke-метрик."""

from types import SimpleNamespace

import pytest
import torch

from uzner.data.melm import MelmExample, preserve_suffix, replace_entity
from uzner.domain import Document, Entity
from uzner.training.melm_generator import masked_batch


def test_unicode_replacement_preserves_other_spans():
    """Новая длина сдвигает последующие offsets, не повреждая Unicode."""
    doc = Document("x", "🙂 Ali va Toshkent", (Entity(2, 5, "NAME"), Entity(9, 17, "GEO")))
    result = replace_entity(doc, 0, "Ўткир", variant="1")
    assert result.text == "🙂 Ўткир va Toshkent"
    assert result.entities == (Entity(2, 7, "NAME"), Entity(11, 19, "GEO"))
    assert result.hash == replace_entity(doc, 0, "Ўткир", variant="1").hash
    assert doc.text == "🙂 Ali va Toshkent"
    with pytest.raises(ValueError):
        replace_entity(doc, 0, "<NAME>", variant="1")


def test_suffix_and_script_guard():
    """Отсекает потерю распознанного окончания и смену письменности."""
    assert preserve_suffix("Toshkentda", "Samarqandda")
    assert not preserve_suffix("Toshkentda", "Samarqand")
    assert not preserve_suffix("Тошкент", "Samarqand")


def test_only_entity_tokens_are_mlm_targets():
    """Контекст, маркеры и padding никогда не становятся MLM-целями."""
    example = MelmExample("x", 0, (0, 8, 9, 10, 8, 2), (2, 3), ((2, 3),))
    ids, mask, labels = masked_batch(
        [example], SimpleNamespace(pad_token_id=1, mask_token_id=7), 0.0
    )
    assert (labels != -100).sum() == 1
    assert torch.equal(labels[0, [0, 1, 4, 5]], torch.full((4,), -100))
    assert ids[0, 1] == 8
    assert mask.all()
