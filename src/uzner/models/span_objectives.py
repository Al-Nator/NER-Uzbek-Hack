"""Дополнительные цели span-моделей без изменения exact-span декодера."""

import torch
from torch.nn import functional

from uzner.models.span_heads import global_pointer_loss


def smooth_boundaries(targets: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Переносит epsilon массы gold на валидных соседей Manhattan-радиуса 1."""
    soft = (targets == 1).float()
    if epsilon == 0:
        return targets.clone()
    # Gold другого span никогда не превращается в соседний отрицательный пример.
    negative = targets == 0
    positive = targets == 1
    degree = torch.zeros_like(soft)
    destinations = []
    for axis in (-2, -1):
        for shift in (-1, 1):
            available = torch.roll(negative, -shift, axis)
            edge = [slice(None)] * 4
            edge[axis] = -1 if shift == 1 else 0
            available[tuple(edge)] = False
            degree += available
            destinations.append((axis, shift, available))
    mass = positive.float() * epsilon * (degree > 0)
    soft -= mass
    for axis, shift, available in destinations:
        soft += torch.roll(mass / degree.clamp_min(1) * available, shift, axis)
    return soft.clamp_max(1).masked_fill(targets < 0, -100)


def smoothed_pointer_loss(
    logits: torch.Tensor, targets: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Обобщает GP logsumexp на веса мягких положительных и отрицательных пар."""
    if epsilon == 0:
        return global_pointer_loss(logits, targets)
    soft = smooth_boundaries(targets, epsilon).flatten(2)
    scores = logits.float().flatten(2)
    valid = soft >= 0
    y = soft.clamp(0, 1)
    positive = (-scores + y.clamp_min(1e-30).log()).masked_fill(~valid | (y == 0), -torch.inf)
    negative = (scores + (1 - y).clamp_min(1e-30).log()).masked_fill(~valid | (y == 1), -torch.inf)
    zero = torch.zeros_like(scores[..., :1])
    return (
        torch.logsumexp(torch.cat((positive, zero), -1), -1)
        + torch.logsumexp(torch.cat((negative, zero), -1), -1)
    ).mean()


def hard_span_loss(logits: torch.Tensor, targets: torch.Tensor, topk: int) -> torch.Tensor:
    """Учит gold, соседние границы, неверные классы и online top-k false positives."""
    positive = targets == 1
    near = positive.any(1, keepdim=True).expand_as(positive).clone()
    for axis in (-2, -1):
        for shift in (-2, -1, 1, 2):
            shifted = torch.roll(positive, shift, axis)
            edge = [slice(None)] * 4
            edge[axis] = slice(0, shift) if shift > 0 else slice(shift, None)
            shifted[tuple(edge)] = False
            near |= shifted
    losses = []
    for score, target, local in zip(logits.float(), targets, near, strict=True):
        negative = target == 0
        selected = local & negative
        values = score.detach().masked_fill(~negative, -torch.inf).flatten()
        n = min(topk, int(negative.sum()))
        if n:
            selected.flatten()[values.topk(n).indices] = True
        terms = []
        if selected.any():
            terms.append(functional.softplus(score[selected]).mean())
        if (target == 1).any():
            terms.append(functional.softplus(-score[target == 1]).mean())
        losses.append(sum(terms) if terms else score.sum() * 0)
    return torch.stack(losses).mean()
