"""Span-модель с тем же encoder-loader и checkpoint-контрактом."""

import torch
from torch import nn
from torch.nn import functional as functional

from uzner.config import ModelConfig
from uzner.domain import LABELS
from uzner.models.span_heads import BiaffineHead, GlobalPointerHead, global_pointer_loss
from uzner.models.token_tagger import TaggerOutput, TokenTagger


class SpanTagger(TokenTagger):
    """Заменяет token head на классификатор пар без BIO/CRF-декодирования."""

    def __init__(self, encoder: nn.Module, config: ModelConfig) -> None:
        """Создаёт только span-head; наследует загрузку и checkpointing encoder-а."""
        nn.Module.__init__(self)
        self.encoder, self.model_config = encoder, config
        self.dropout = nn.Dropout(config.dropout)
        head_type = BiaffineHead if config.head == "biaffine" else GlobalPointerHead
        self.classifier = head_type(encoder.config.hidden_size, config.span_head_size, len(LABELS))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> TaggerOutput:
        """Считает scores пар и CE или GlobalPointer loss по span-targets."""
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**inputs).last_hidden_state
        logits = self.classifier(self.dropout(hidden))
        loss = None
        if labels is not None:
            if self.model_config.head == "global_pointer":
                loss = global_pointer_loss(logits, labels)
            else:
                # Класс 0 означает NONE; -100 исключает непредставимые пары.
                valid = (labels >= 0).any(dim=1)
                target = torch.zeros_like(labels[:, 0], dtype=torch.long)
                for index in range(len(LABELS)):
                    target[labels[:, index] == 1] = index + 1
                loss = (
                    functional.cross_entropy(logits.float(), target.masked_fill(~valid, -100))
                    if valid.any()
                    else logits.float().sum() * 0
                )
        return TaggerOutput(logits, loss)
