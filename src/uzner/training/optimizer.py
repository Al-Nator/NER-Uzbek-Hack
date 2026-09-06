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
    parameters = model.parameters()
    if config.head_learning_rate is not None:
        encoder = [p for name, p in model.named_parameters() if name.startswith("encoder.")]
        head = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
        if not encoder or not head:
            raise ValueError("Раздельный LR требует непустые encoder и head")
        parameters = [
            {"params": encoder, "lr": config.learning_rate, "name": "encoder"},
            {"params": head, "lr": config.head_learning_rate, "name": "head"},
        ]
    return optimizer_type(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
