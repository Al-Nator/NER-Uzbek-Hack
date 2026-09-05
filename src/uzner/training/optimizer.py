"""Единое создание AdamW для обучения и проверки вместимости GPU."""

from torch import nn
from torch.optim import AdamW, Optimizer

from uzner.config import TrainingConfig


def make_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """Выбирает явно заданную точность состояний, не квантуя веса модели."""
    optimizer_type = AdamW
    if config.optimizer == "adamw_8bit":
        from bitsandbytes.optim import AdamW8bit

        optimizer_type = AdamW8bit
    return optimizer_type(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
