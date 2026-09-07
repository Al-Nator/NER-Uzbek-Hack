"""Линейная CRF с жёсткими BIO/BIOES-ограничениями."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from uzner.data.tagging import TagScheme
from uzner.models.transitions import allowed_end, allowed_start, allowed_transition

INVALID_SCORE = -10_000.0


def viterbi_path(
    emissions: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    transitions: torch.Tensor,
) -> list[int]:
    """Считает общий Viterbi-путь; та же функция допускает TorchScript на CPU."""
    score = start + emissions[0]
    backpointers: list[torch.Tensor] = []
    for index in range(1, emissions.size(0)):
        candidates = score[:, None] + transitions
        score, pointers = candidates.max(dim=0)
        score = score + emissions[index]
        backpointers.append(pointers)
    final = int((score + end).argmax().item())
    path = [final]
    for index in range(len(backpointers) - 1, -1, -1):
        path.append(int(backpointers[index][path[-1]].item()))
    path.reverse()
    return path


class LinearChainCrf(nn.Module):
    """Обучаемая linear-chain CRF для одной схемы тегов."""

    def __init__(self, tags: Sequence[str], scheme: TagScheme) -> None:
        """Создаёт параметры переходов и маски валидных путей."""
        super().__init__()
        if not tags:
            raise ValueError("CRF требует непустой словарь тегов")
        self.tags = tuple(tags)
        self.scheme = scheme
        tag_count = len(self.tags)
        self.start_transitions = nn.Parameter(torch.empty(tag_count))
        self.end_transitions = nn.Parameter(torch.empty(tag_count))
        self.transitions = nn.Parameter(torch.empty(tag_count, tag_count))
        self.register_buffer(
            "valid_start",
            torch.tensor([allowed_start(tag, scheme) for tag in self.tags]),
        )
        self.register_buffer(
            "valid_end",
            torch.tensor([allowed_end(tag, scheme) for tag in self.tags]),
        )
        self.register_buffer(
            "valid_transitions",
            torch.tensor(
                [
                    [allowed_transition(previous, current, scheme) for current in self.tags]
                    for previous in self.tags
                ]
            ),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Инициализирует переходы малыми случайными значениями."""
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _masked_start(self) -> torch.Tensor:
        """Возвращает start-оценки с запретами схемы."""
        return self.start_transitions.masked_fill(~self.valid_start, INVALID_SCORE)

    def _masked_end(self) -> torch.Tensor:
        """Возвращает end-оценки с запретами схемы."""
        return self.end_transitions.masked_fill(~self.valid_end, INVALID_SCORE)

    def _masked_transitions(self) -> torch.Tensor:
        """Возвращает матрицу переходов с запретами схемы."""
        return self.transitions.masked_fill(~self.valid_transitions, INVALID_SCORE)

    def _gold_score(self, emissions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Считает score одной gold-последовательности."""
        if labels.numel() == 0:
            raise ValueError("CRF не может обучаться на пустой последовательности")
        score = self._masked_start()[labels[0]] + emissions[0, labels[0]]
        transitions = self._masked_transitions()
        for index in range(1, labels.numel()):
            score = score + transitions[labels[index - 1], labels[index]]
            score = score + emissions[index, labels[index]]
        return score + self._masked_end()[labels[-1]]

    def _log_partition(self, emissions: torch.Tensor) -> torch.Tensor:
        """Считает log-partition одной последовательности."""
        score = self._masked_start() + emissions[0]
        transitions = self._masked_transitions()
        for emission in emissions[1:]:
            candidates = score[:, None] + transitions + emission[None, :]
            score = torch.logsumexp(candidates, dim=0)
        return torch.logsumexp(score + self._masked_end(), dim=0)

    def neg_log_likelihood(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Усредняет negative log-likelihood по непустым окнам batch."""
        if logits.ndim != 3 or labels.shape != logits.shape[:2]:
            raise ValueError("Несовместимые размерности logits и labels")
        losses: list[torch.Tensor] = []
        for row_logits, row_labels in zip(logits.float(), labels, strict=True):
            mask = row_labels >= 0
            if not mask.any():
                continue
            emissions = row_logits[mask]
            gold = row_labels[mask]
            losses.append(self._log_partition(emissions) - self._gold_score(emissions, gold))
        if not losses:
            raise ValueError("Batch не содержит обучаемых токенов")
        return torch.stack(losses).mean()

    def decode_one(self, emissions: torch.Tensor) -> tuple[int, ...]:
        """Декодирует лучший валидный путь для одной последовательности."""
        if emissions.ndim != 2 or emissions.shape[0] == 0:
            raise ValueError("emissions должны иметь форму nonempty sequence x tags")
        decoder = getattr(self, "_compiled_decode", viterbi_path)
        return tuple(
            decoder(
                emissions.float(),
                self._masked_start(),
                self._masked_end(),
                self._masked_transitions(),
            )
        )

    def compile_cpu_decode(self) -> None:
        """Компилирует только общий Viterbi-цикл, не меняя веса или арифметику."""
        if self.transitions.device.type != "cpu":
            raise ValueError("Компилируемый декодер предназначен для CPU")
        self._compiled_decode = torch.jit.script(viterbi_path)
