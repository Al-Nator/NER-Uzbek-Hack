"""Регрессии новых целей: маски, градиенты, BIOES и reference-паритет."""

from dataclasses import replace

import pytest
import torch

from uzner.config import ModelConfig
from uzner.data.tagging import build_tag_vocabulary
from uzner.models.crf import LinearChainCrf
from uzner.models.span_auxiliary import SpanAuxiliary, auxiliary_targets, batched_crf_loss
from uzner.models.span_heads import global_pointer_loss
from uzner.models.span_objectives import hard_span_loss, smooth_boundaries, smoothed_pointer_loss
from uzner.research_config import SpanResearchConfig


def targets() -> torch.Tensor:
    """Создаёт batch с одним положительным span и специальными токенами."""
    t = torch.full((2, 3, 7, 7), -100.0)
    for i in range(1, 6):
        t[:, :, i, i:6] = 0
    t[0, 0, 2, 3] = 1
    t[1, 1, 1, 1] = 1
    return t


def test_smoothing_mass_masks_and_zero_parity() -> None:
    """Сглаживание сохраняет массу, маски и точную цель при epsilon=0."""
    t = targets()
    y = smooth_boundaries(t, 0.1)
    assert torch.equal(y < 0, t < 0)
    assert y[y >= 0].sum().item() == pytest.approx(2)
    assert y[0, 0, 2, 3].item() == pytest.approx(0.9)
    z = torch.randn_like(t, requires_grad=True)
    assert torch.equal(global_pointer_loss(z, t), smoothed_pointer_loss(z, t, 0))
    loss = smoothed_pointer_loss(z, t, 0.1)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert (z.grad[t < 0] == 0).all()


def test_auxiliary_targets_and_batched_crf_parity() -> None:
    """BIOES отражает spans, а пакетная CRF совпадает с исходной реализацией."""
    t = auxiliary_targets(targets())
    assert t.tags[0].tolist() == [-100, 0, 1, 3, 0, 0, -100]
    assert t.tags[1].tolist() == [-100, 8, 0, 0, 0, 0, -100]
    crf = LinearChainCrf(build_tag_vocabulary("bioes"), "bioes")
    z = torch.randn(2, 7, 13, requires_grad=True)
    expected = torch.stack(
        [
            crf.neg_log_likelihood(e[None], tag[None]) / (tag >= 0).sum()
            for e, tag in zip(z, t.tags, strict=True)
        ]
    ).mean()
    actual = batched_crf_loss(crf, z, t.tags)
    assert torch.allclose(expected, actual, atol=1e-5)
    actual.backward()
    assert torch.isfinite(z.grad).all()


def test_auxiliary_heads_receive_gradients() -> None:
    """Обе головы и общий hidden получают конечные градиенты."""
    module = SpanAuxiliary(8, 3, SpanResearchConfig(bioes_crf_weight=0.25, boundary_weight=0.25))
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    losses = module(hidden, targets())
    sum(losses.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    assert torch.isfinite(hidden.grad).all()


def test_hard_negatives_respect_gold_and_ignored() -> None:
    """Hard-negative loss повышает gold-score и снижает ложные scores, игнорируя padding."""
    t = targets()
    z = torch.zeros_like(t, requires_grad=True)
    hard_span_loss(z, t, 5).backward()
    assert (z.grad[t == 1] < 0).all()
    assert (z.grad[t == 0] >= 0).all()
    assert (z.grad[t < 0] == 0).all()


def test_empty_masks_have_finite_zero_gradient() -> None:
    """Окна без supervision не создают NaN ни в одной цели."""
    t = torch.full((2, 3, 4, 4), -100.0)
    z = torch.randn_like(t, requires_grad=True)
    loss = smoothed_pointer_loss(z, t, 0.1) + hard_span_loss(z, t, 3)
    loss.backward()
    assert loss.item() == 0 and (z.grad == 0).all()


def test_config_rejects_wrong_architecture() -> None:
    """Новые цели нельзя молча включить для несовместимой архитектуры."""
    c = ModelConfig("token_tagging", "bio", "softmax", "greedy")
    with pytest.raises(ValueError):
        replace(c, research=SpanResearchConfig(smoothing=0.1))
