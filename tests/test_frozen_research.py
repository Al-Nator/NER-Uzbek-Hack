"""Контракты kNN, символов и отсутствия gold при инференсе."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional

from uzner.domain import Document, Entity, Prediction
from uzner.models.char_boundary import CharBoundaryHead, CharConfig, character_window
from uzner.models.memory_knn import KnnConfig, TokenMemory, interpolate_spans
from uzner.training.char_features import (
    BoundaryExample,
    apply_boundaries,
    boundary_batch,
    collect_boundaries,
)
from uzner.training.memory_pipeline import build_memory


def test_blockwise_knn_matches_dense():
    """Блочный поиск совпадает с полным cosine top-k и сохраняет фон O."""
    torch.manual_seed(12)
    keys = functional.normalize(torch.randn(17, 4), dim=-1)
    labels = torch.arange(17) % 13
    groups = torch.arange(17) % 3
    memory = TokenMemory(keys, labels, groups)
    queries = torch.randn(5, 4)
    config = KnnConfig(neighbors=4, block_size=3)
    actual, valid = memory.posterior(queries, 1, config)
    scores = functional.normalize(queries, dim=-1) @ keys.T
    scores[:, groups == 1] = -torch.inf
    values, ids = scores.topk(4, dim=1)
    expected = torch.zeros_like(actual).scatter_add_(1, labels[ids], (values / 0.1).softmax(1))
    assert torch.allclose(actual, expected)
    assert valid.all()
    assert torch.allclose(actual.sum(1), torch.ones(5))


def test_memory_empty_and_self_match_fallback():
    """Собственный документ и отсутствие памяти не дают ложной уверенности."""
    memory = TokenMemory(torch.ones(1, 2), torch.tensor([4]), torch.tensor([7]))
    posterior, valid = memory.posterior(torch.ones(2, 2), 7, KnnConfig())
    assert not valid.any() and not posterior.any()
    scores = torch.rand(3, 2, 2)
    assert torch.equal(interpolate_spans(scores, posterior, valid, 0.2), scores)
    empty = TokenMemory(torch.empty(0, 2), torch.empty(0, dtype=torch.long), torch.empty(0))
    assert not empty.posterior(torch.ones(1, 2), -1, KnnConfig())[1].any()


def test_memory_o_vote_and_singleton_label():
    """O уменьшает span-score; однотокенная сущность требует S, а не B/E."""
    scores = torch.ones(3, 2, 2)
    posterior = torch.zeros(2, 13)
    posterior[:, 0] = 1
    valid = torch.ones(2, dtype=torch.bool)
    assert torch.allclose(interpolate_spans(scores, posterior, valid, 0.2), scores * 0.8)
    assert interpolate_spans(scores, posterior, valid, 0) is scores
    posterior.zero_()
    posterior[:, 1] = 0.5
    posterior[:, 3] = 0.5
    result = interpolate_spans(scores, posterior, valid, 0.2)
    assert result[0, 0, 0] == 0.8
    assert float(result[0, 0, 1]) == pytest.approx(0.9)


@pytest.mark.parametrize(
    "config,kwargs",
    [
        (KnnConfig, {"alpha": 2}),
        (KnnConfig, {"temperature": 0}),
        (KnnConfig, {"capacity": 0}),
        (KnnConfig, {"sampling_rate": 0}),
        (CharConfig, {"radius": 0}),
        (CharConfig, {"confidence": 2}),
    ],
)
def test_invalid_research_parameters(config, kwargs):
    """Некорректные параметры отвергаются до начала опыта."""
    with pytest.raises(ValueError):
        config(**kwargs)


def test_character_head_unicode_and_checkpoint(tmp_path):
    """Char-head имеет конечные градиенты и восстанавливается без encoder-а."""
    config = CharConfig()
    alphabet = {"🙂": 2, "Ў": 3, "a": 4}
    chars, valid = character_window("🙂Ўa", 0, alphabet, config)
    assert chars[10:13] == (2, 3, 4)
    assert valid == (False, False, False, False, True, True, True, True, False)
    model = CharBoundaryHead(8, 5, config)
    hidden = torch.randn(1, 8)
    inputs = (hidden, torch.tensor([chars]), torch.tensor([0]), torch.tensor([valid]))
    logits = model(*inputs)
    functional.cross_entropy(logits, torch.tensor([5])).backward()
    assert torch.isfinite(model.token.weight.grad).all()
    assert logits[0, 0] < -1000
    torch.save(model.state_dict(), tmp_path / "head.pt")
    other = CharBoundaryHead(8, 5, config)
    other.load_state_dict(torch.load(tmp_path / "head.pt", weights_only=True))
    assert torch.equal(model(*inputs), other(*inputs))


def test_character_edits_are_offsets_not_normalization():
    """Возвращает Unicode-границу внутри токена; невалидная правка откатывается."""
    config = CharConfig()
    doc = Document("d", "🙂 Ali-da")
    prediction = Prediction("d", (Entity(2, 8, "NAME", 0.9),))
    example = BoundaryExample(0, 0, 1, 8, torch.zeros(8), (), (), 3)
    output = apply_boundaries((prediction,), [example], [1], [0.9], (doc,), config)
    assert output[0].entities == (Entity(2, 5, "NAME", 0.9),)
    assert apply_boundaries((prediction,), [example], [1], [0.5], (doc,), config) == (prediction,)
    invalid = replace(example, anchor=0)
    assert apply_boundaries((prediction,), [invalid], [0], [0.9], (doc,), config) == (prediction,)


class FakePredictor:
    """Возвращает синтетические окна и фиксирует реально переданные документы."""

    def windows(self, documents, *, with_labels=False):
        """Имитирует два одинаковых окна для проверки дедупликации."""
        self.documents = documents
        for index, _doc in enumerate(documents):
            feature = SimpleNamespace(
                offsets=((0, 0), (0, 3), (4, 7)), labels=(-100, 0, 8), window_index=0
            )
            window = SimpleNamespace(document_index=index, feature=feature, hidden=torch.ones(3, 8))
            yield window
            yield window


def test_memory_builder_excludes_dev_and_duplicate_windows(tmp_path):
    """Ни dev-документ, ни повтор перекрытого токена не попадают в память."""
    predictor = FakePredictor()
    train = (Document("a", "Ali Vali"), Document("b", "dev text"))
    run = SimpleNamespace(root=tmp_path, log=lambda _values: None)
    memory = build_memory(predictor, train, {"dev text"}, KnnConfig(sampling_rate=1), run)
    assert [d.hash for d in predictor.documents] == ["a"]
    assert len(memory.keys) == 2
    assert sorted(memory.labels.tolist()) == [0, 8]


def test_character_inference_does_not_use_gold():
    """Изменение gold не меняет признаки кандидатов при training=False."""
    predictor = FakePredictor()
    config = CharConfig()
    a = Document("x", "Ali Vali", (Entity(0, 3, "NAME"),))
    b = Document("x", "Ali Vali", (Entity(4, 7, "ORG"),))
    predictions = (Prediction("x", (Entity(0, 3, "GEO", 0.8),)),)
    first = collect_boundaries(predictor, (a,), {}, config, training=False, predictions=predictions)
    second = collect_boundaries(
        predictor, (b,), {}, config, training=False, predictions=predictions
    )
    assert len(first) == len(second) == 2
    assert all(e.target == -100 for e in first)
    assert [(e.kind, e.anchor, e.chars) for e in first] == [
        (e.kind, e.anchor, e.chars) for e in second
    ]
    assert boundary_batch(first, torch.device("cpu"))[0].shape == (2, 8)
    with pytest.raises(ValueError):
        collect_boundaries(predictor, (a,), {}, config, training=False)
