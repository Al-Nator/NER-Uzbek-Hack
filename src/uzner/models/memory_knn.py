"""Контекстная train-only память BIOES с фоном O и блочным точным top-k."""

from dataclasses import dataclass

import torch
from torch.nn import functional


@dataclass(frozen=True)
class KnnConfig:
    """Фиксированные параметры памяти, не подбираемые по dev."""

    capacity: int = 100_000
    neighbors: int = 16
    temperature: float = 0.1
    alpha: float = 0.2
    block_size: int = 4096
    sampling_rate: float = 0.06
    seed: int = 42

    def __post_init__(self) -> None:
        """Проверяет пределы параметров до выделения памяти."""
        if min(self.capacity, self.neighbors, self.block_size) < 1:
            raise ValueError("Размеры памяти и top-k должны быть положительными")
        if self.temperature <= 0 or not 0 <= self.alpha <= 1:
            raise ValueError("Некорректная температура или вес памяти")
        if not 0 < self.sampling_rate <= 1:
            raise ValueError("Некорректная доля случайной выборки")


@dataclass
class TokenMemory:
    """Нормированные ключи, BIOES-метки и группы исходных текстов."""

    keys: torch.Tensor
    labels: torch.Tensor
    groups: torch.Tensor

    def posterior(
        self, queries: torch.Tensor, group: int, config: KnnConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ищет соседей без собственного документа; отсутствие памяти явно отмечается."""
        n = len(queries)
        result = queries.new_zeros((n, 13), dtype=torch.float32)
        if not len(self.keys) or not n:
            return result, torch.zeros(n, dtype=torch.bool, device=queries.device)
        k = min(config.neighbors, len(self.keys))
        queries = functional.normalize(queries.float(), dim=-1).to(self.keys.dtype)
        best = queries.new_full((n, k), -torch.inf)
        indices = torch.zeros((n, k), dtype=torch.long, device=queries.device)
        for offset in range(0, len(self.keys), config.block_size):
            keys = self.keys[offset : offset + config.block_size].to(queries.device)
            scores = queries @ keys.T
            groups = self.groups[offset : offset + len(keys)].to(queries.device)
            scores[:, groups == group] = -torch.inf
            ids = torch.arange(offset, offset + len(keys), device=queries.device).expand(n, -1)
            values, chosen = torch.cat((best, scores), 1).topk(k, dim=1)
            indices = torch.cat((indices, ids), 1).gather(1, chosen)
            best = values
        valid = torch.isfinite(best).any(1)
        weights = (best.float() / config.temperature).masked_fill(~valid[:, None], 0).softmax(1)
        weights *= valid[:, None]
        labels = self.labels.to(queries.device)[indices]
        result.scatter_add_(1, labels, weights)
        return result, valid


def interpolate_spans(
    probabilities: torch.Tensor, posterior: torch.Tensor, valid: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Добавляет голос начала/конца нужного класса, включая однотокенный S."""
    if alpha == 0:
        return probabilities
    result = probabilities.clone()
    for label in range(3):
        b, _i, e, s = range(1 + label * 4, 5 + label * 4)
        start = posterior[:, b] + posterior[:, s]
        end = posterior[:, e] + posterior[:, s]
        score = (start[:, None] * end[None, :]).sqrt()
        # Для одного токена допустим только S, а не независимые B/E.
        score.diagonal().copy_(posterior[:, s])
        known = valid[:, None] & valid[None, :]
        result[label] = torch.where(
            known, (1 - alpha) * probabilities[label] + alpha * score, probabilities[label]
        )
    return result
