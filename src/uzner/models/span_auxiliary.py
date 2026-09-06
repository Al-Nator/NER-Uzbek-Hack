"""BIOES/CRF и START/END как обучающие сигналы общего encoder-а."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from uzner.data.tagging import build_tag_vocabulary
from uzner.models.crf import LinearChainCrf
from uzner.research_config import SpanResearchConfig


@dataclass(frozen=True)
class AuxiliaryTargets:
    """Цели последовательности и отдельных границ с масками неизвестных позиций."""

    tags: torch.Tensor
    starts: torch.Tensor
    ends: torch.Tensor
    start_mask: torch.Tensor
    end_mask: torch.Tensor


def auxiliary_targets(labels: torch.Tensor) -> AuxiliaryTargets:
    """Выводит BIOES и границы только из представимых gold-пар окна."""
    valid = (labels >= 0).any(1)
    sm, em = valid.any(-1), valid.any(-2)
    tags = torch.zeros_like(sm, dtype=torch.long).masked_fill(~(sm | em), -100)
    starts, ends = (labels == 1).any(-1), (labels == 1).any(-2)
    for row, label, start, end in (labels == 1).nonzero().tolist():
        indices = torch.arange(start, end + 1, device=labels.device)
        indices = indices[tags[row, indices] >= 0]
        if not len(indices):
            continue
        if (tags[row, indices] > 0).any():
            raise ValueError("Пересекающиеся span targets нельзя преобразовать в BIOES")
        tags[row, indices] = 2 + 4 * label
        tags[row, indices[0]] = 1 + 4 * label
        tags[row, indices[-1]] = 3 + 4 * label
        if len(indices) == 1:
            tags[row, indices[0]] = 4 + 4 * label
    return AuxiliaryTargets(tags, starts.float(), ends.float(), sm, em)


def batched_crf_loss(
    crf: LinearChainCrf, emissions: torch.Tensor, tags: torch.Tensor
) -> torch.Tensor:
    """Считает ту же constrained CRF-NLL, векторизуя batch и нормируя на токены."""
    sequences = [e[t >= 0].float() for e, t in zip(emissions, tags, strict=True) if (t >= 0).any()]
    golds = [t[t >= 0] for t in tags if (t >= 0).any()]
    if not sequences:
        return emissions.float().sum() * 0
    lengths = torch.tensor([len(t) for t in golds], device=emissions.device)
    e = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    t = nn.utils.rnn.pad_sequence(golds, batch_first=True)
    transition = crf._masked_transitions()
    score = crf._masked_start()[None] + e[:, 0]
    gold_score = crf._masked_start()[t[:, 0]] + e[:, 0].gather(1, t[:, :1]).squeeze(1)
    for i in range(1, e.shape[1]):
        active = lengths > i
        updated = torch.logsumexp(score[:, :, None] + transition[None], 1) + e[:, i]
        score = torch.where(active[:, None], updated, score)
        gold_score += (
            transition[t[:, i - 1], t[:, i]] + e[:, i].gather(1, t[:, i : i + 1]).squeeze(1)
        ) * active
    gold_score += crf._masked_end()[t.gather(1, (lengths - 1)[:, None]).squeeze(1)]
    partition = torch.logsumexp(score + crf._masked_end()[None], -1)
    return ((partition - gold_score) / lengths).mean()


class SpanAuxiliary(nn.Module):
    """Обучает дополнительные головы; не меняет основной GP-декодер."""

    def __init__(self, size: int, classes: int, config: SpanResearchConfig) -> None:
        """Создаёт только включённые конфигом головы."""
        super().__init__()
        self.config = config
        if config.bioes_crf_weight:
            self.sequence = nn.Linear(size, 1 + 4 * classes)
            self.crf = LinearChainCrf(build_tag_vocabulary("bioes"), "bioes")
        if config.boundary_weight:
            self.boundaries = nn.Linear(size, 2 * classes)

    def forward(self, hidden: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        """Возвращает ненормированные по весам компоненты дополнительных losses."""
        if not (self.config.bioes_crf_weight or self.config.boundary_weight):
            return {}
        target = auxiliary_targets(labels)
        losses = {}
        if self.config.bioes_crf_weight:
            losses["bioes_crf"] = batched_crf_loss(self.crf, self.sequence(hidden), target.tags)
        if self.config.boundary_weight:
            start, end = self.boundaries(hidden).float().transpose(1, 2).chunk(2, 1)
            terms = []
            for scores, gold, mask in (
                (start, target.starts, target.start_mask),
                (end, target.ends, target.end_mask),
            ):
                valid = mask[:, None, :].expand_as(gold)
                terms.append(
                    functional.binary_cross_entropy_with_logits(scores[valid], gold[valid])
                    if valid.any()
                    else scores.sum() * 0
                )
            losses["boundary"] = torch.stack(terms).mean()
        return losses
