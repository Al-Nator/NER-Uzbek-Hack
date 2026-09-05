"""Span targets, две головы, Unicode decoding и отсутствие gold leakage."""

from dataclasses import replace

import pytest
import torch

from tests.helpers import FakeTokenizer, TinyEncoder
from uzner.config import ModelConfig, TokenizationConfig, TrainingConfig
from uzner.data.spans import SpanCollator, span_coverage, span_pair_mask, with_span_targets
from uzner.data.windows import WindowFeature, build_window_features
from uzner.domain import Document, Entity
from uzner.models.span_heads import BiaffineHead, global_pointer_loss, rotary_positions
from uzner.models.span_tagger import SpanTagger
from uzner.training.data_setup import make_loader
from uzner.training.inference import run_inference
from uzner.training.span_prediction import (
    SpanPredictionAccumulator,
    SpanWindowScores,
    decode_span_windows,
    select_flat_entities,
)


def _feature(offsets) -> WindowFeature:
    """Создаёт окно с явными offsets, включая нестандартные границы."""
    return WindowFeature(0, 0, (1,) * len(offsets), (1,) * len(offsets), None, offsets, None)


def test_exact_targets_unicode_duplicate_offsets_and_padding() -> None:
    """Не теряет Unicode и объединяет повторные offsets без ложных singleton-ов."""
    document = Document("u", "Ўз 😀 Ali", (Entity(0, 2, "ORG"), Entity(5, 8, "NAME")))
    feature = _feature(((0, 0), (0, 2), (0, 2), (3, 4), (5, 8), (0, 0)))
    features = with_span_targets((feature,), (document,))
    labels = SpanCollator(0)(features).labels
    assert labels[0, 0, 1, 2] == 1
    assert labels[0, 1, 4, 4] == 1
    assert labels[0, 0, 1, 1] == -100
    assert (labels[:, :, 0, :] == -100).all()
    assert not span_pair_mask(feature.offsets, 9)[6:].any()
    assert span_coverage(features, (document,))["recall_ceiling"] == 1


def test_partial_unrepresentable_gold_and_long_spans() -> None:
    """Исключает ложные negatives на стыках и не обрезает длинные сущности."""
    document = Document("d", "abc def gh", (Entity(1, 3, "NAME"), Entity(4, 10, "ORG")))
    feature = _feature(((0, 3), (4, 7)))
    features = with_span_targets((feature,), (document,))
    assert (SpanCollator(0)(features).labels == -100).all()
    assert span_coverage(features, (document,))["represented"] == 0
    text = " ".join(["word"] * 160)
    long_doc = Document("long", text, (Entity(0, len(text), "ORG"),))
    windows = build_window_features(
        (long_doc,), FakeTokenizer(), TokenizationConfig(256, 32), "bioes", with_labels=False
    )
    long_features = with_span_targets(windows, (long_doc,))
    assert long_features[0].span_targets == ((0, 1, 160),)
    assert span_coverage(long_features, (long_doc,))["represented"] == 1


@pytest.mark.parametrize("head", ["biaffine", "global_pointer"])
def test_span_heads_backward_masking_empty_and_inference(head) -> None:
    """Обе головы дают конечный loss/градиент, а masked targets не влияют на loss."""
    config = ModelConfig("span", "bioes", head, "span", dropout=0)
    model = SpanTagger(TinyEncoder(), config)
    ids = torch.ones((2, 5), dtype=torch.long)
    labels = torch.zeros((2, 3, 5, 5))
    labels[:, :, 0] = -100
    labels[:, 1, 1, 3] = 1
    output = model(ids, ids, labels=labels)
    assert output.logits.shape == (2, 4 if head == "biaffine" else 3, 5, 5)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.encoder.embedding.weight.grad.abs().sum() > 0
    assert model(ids, ids).loss is None
    empty = model(ids, ids, labels=torch.full_like(labels, -100))
    assert empty.loss == 0
    empty.loss.backward()


def test_global_pointer_loss_and_rope_math() -> None:
    """Сверяет loss с аналитикой, RoPE-нормы и относительные позиции."""
    logits = torch.tensor([[[[0.0, 2.0, 1000.0]]]], requires_grad=True)
    targets = torch.tensor([[[[0.0, 1.0, -100.0]]]])
    loss = global_pointer_loss(logits, targets)
    assert loss.item() == pytest.approx(
        torch.log(torch.tensor(2.0)).item()
        + torch.nn.functional.softplus(torch.tensor(-2.0)).item()
    )
    loss.backward()
    assert logits.grad[0, 0, 0, 2] == 0
    assert logits.grad[0, 0, 0, 1] < 0 < logits.grad[0, 0, 0, 0]
    value = torch.randn(1, 6, 3, 8)
    rotated = rotary_positions(value)
    assert torch.allclose(value.norm(dim=-1), rotated.norm(dim=-1), atol=1e-6)
    assert torch.equal(rotated[:, 0], value[:, 0])


def test_biaffine_math() -> None:
    """Проверяет билинейный член и bias явным матричным произведением."""
    head = BiaffineHead(8, 4, 3)
    hidden = torch.randn(1, 2, 8)
    left = torch.nn.functional.pad(head.start(hidden), (0, 1), value=1)
    right = torch.nn.functional.pad(head.end(hidden), (0, 1), value=1)
    assert torch.allclose(head(hidden)[0, 2, 0, 1], left[0, 0] @ head.weight[2] @ right[0, 1])


def test_window_merging_includes_negative_votes_and_flat_contract() -> None:
    """Не завышает score фильтрацией negatives до усреднения; выдаёт плоские spans."""
    offsets = ((0, 0), (0, 2), (3, 4), (0, 0))
    first = torch.zeros(3, 4, 4)
    second = first.clone()
    first[0, 1, 2], second[0, 1, 2] = 0.9, 0.05
    first[1, 1, 1], second[1, 1, 1] = 0.8, 0.9
    result = decode_span_windows(
        "x", [SpanWindowScores(offsets, first), SpanWindowScores(offsets, second)], 0.5
    )
    assert len(result.entities) == 1
    assert result.entities[0].to_mapping() == {"label": "NAME", "start": 0, "end": 2}
    assert result.entities[0].score == pytest.approx(0.85)
    assert select_flat_entities(
        [Entity(0, 3, "ORG", 0.7), Entity(0, 2, "NAME", 0.8), Entity(2, 4, "GEO", 0.8)]
    ) == (Entity(0, 2, "NAME", 0.8), Entity(2, 4, "GEO", 0.8))
    accumulator = SpanPredictionAccumulator(0.5, ("empty", "x"), [], [])
    accumulator.add(1, SpanWindowScores(offsets, first))
    with pytest.raises(ValueError, match="порядка"):
        accumulator.add(0, SpanWindowScores(offsets, first))
    assert [p.hash for p in accumulator.finish()] == ["empty", "x"]


def test_span_and_optimizer_config_validation() -> None:
    """Отклоняет неописанные комбинации головы, threshold и optimizer."""
    for kwargs in (
        {"head": "softmax"},
        {"decoder": "crf"},
        {"span_head_size": 3},
        {"span_threshold": 0},
        {"dropout": 1},
    ):
        with pytest.raises(ValueError):
            replace(ModelConfig("span", "bioes", "biaffine", "span"), **kwargs)
    with pytest.raises(ValueError, match="optimizer"):
        TrainingConfig(optimizer="unknown")


@pytest.mark.parametrize("head", ["biaffine", "global_pointer"])
def test_inference_is_independent_of_gold(head) -> None:
    """Меняет gold и loss masks, сохраняя совершенно одинаковые predictions."""
    text = "Али Toshkentga"
    gold = Document("x", text, (Entity(0, 3, "NAME"),))
    empty = Document("x", text)
    model = SpanTagger(TinyEncoder(), ModelConfig("span", "bioes", head, "span"))
    tokenizer = FakeTokenizer()
    results = []
    for document in (gold, empty):
        features = with_span_targets(
            build_window_features(
                (document,), tokenizer, TokenizationConfig(16, 4), "bioes", with_labels=False
            ),
            (document,),
        )
        loader = make_loader(
            features, tokenizer, batch_size=1, shuffle=False, seed=42, num_workers=0
        )
        results.append(
            run_inference(
                model, loader, features, (document,), (), torch.device("cpu"), bf16=False
            ).predictions
        )
    assert results[0] == results[1]


@pytest.mark.parametrize("head", ["biaffine", "global_pointer"])
def test_heads_can_overfit_positive_and_negative_pairs(head) -> None:
    """Проверяет обучаемость сущностей, а не только снижение loss на NONE."""
    torch.manual_seed(42)
    model = SpanTagger(TinyEncoder(), ModelConfig("span", "bioes", head, "span", dropout=0))
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(ids)
    labels = torch.zeros(1, 3, 4, 4)
    labels.masked_fill_(~torch.ones(4, 4, dtype=torch.bool).triu(), -100)
    labels[0, 0, 0, 1], labels[0, 1, 2, 2], labels[0, 2, 3, 3] = 1, 1, 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _step in range(100):
        optimizer.zero_grad()
        output = model(ids, mask, labels=labels)
        output.loss.backward()
        optimizer.step()
    logits = model(ids, mask).logits
    scores = logits.softmax(1)[:, 1:] if head == "biaffine" else logits.sigmoid()
    assert torch.equal((scores[labels >= 0] > 0.5), labels[labels >= 0].bool())
