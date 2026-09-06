"""Span-модель с тем же encoder-loader и checkpoint-контрактом."""

import torch
from torch import nn
from torch.nn import functional as functional

from uzner.config import ModelConfig
from uzner.domain import LABELS
from uzner.models.span_auxiliary import SpanAuxiliary
from uzner.models.span_heads import BiaffineHead, GlobalPointerHead
from uzner.models.span_objectives import hard_span_loss, smoothed_pointer_loss
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
        self.auxiliary = SpanAuxiliary(encoder.config.hidden_size, len(LABELS), config.research)

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
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)
        loss = None
        components = {}
        if labels is not None:
            if self.model_config.head == "global_pointer":
                research = self.model_config.research
                loss = smoothed_pointer_loss(logits, labels, research.smoothing)
                components["gp"] = loss
                components.update(self.auxiliary(hidden, labels))
                if research.hard_negative_weight:
                    components["hard_negative"] = hard_span_loss(
                        logits, labels, research.hard_negative_topk
                    )
                for name in ("bioes_crf", "boundary", "hard_negative"):
                    if name in components:
                        loss = loss + getattr(research, name + "_weight") * components[name]
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
        return TaggerOutput(logits, loss, components)
