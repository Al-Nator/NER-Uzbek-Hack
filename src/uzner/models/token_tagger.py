"""Единая token-tagging модель для softmax, constraints и CRF."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional
from transformers import AutoModel

from uzner.config import EncoderConfig, ModelConfig
from uzner.data.tagging import build_tag_vocabulary
from uzner.models.crf import LinearChainCrf
from uzner.models.transitions import constrained_viterbi, split_tag


@dataclass(frozen=True, slots=True)
class TaggerOutput:
    """Выход модели с обязательными emissions и необязательным loss."""

    logits: torch.Tensor
    loss: torch.Tensor | None


@contextmanager
def _safetensors_conversion_disabled(disabled: bool) -> Iterator[None]:
    """Временно запрещает фоновую конверсию bin-весов на Hugging Face."""
    variable = "DISABLE_SAFETENSORS_CONVERSION"
    previous = os.environ.get(variable)
    if disabled:
        os.environ[variable] = "1"
    try:
        yield
    finally:
        if disabled:
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous


class TokenTagger(nn.Module):
    """Добавляет проекцию и optional CRF поверх pretrained encoder-а."""

    def __init__(self, encoder: nn.Module, config: ModelConfig) -> None:
        """Создаёт голову для выбранной схемы тегов."""
        super().__init__()
        hidden_size = getattr(getattr(encoder, "config", None), "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("Encoder config не содержит корректный hidden_size")
        self.encoder = encoder
        self.model_config = config
        self.tags = build_tag_vocabulary(config.tag_scheme)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(hidden_size, len(self.tags))
        self.crf = LinearChainCrf(self.tags, config.tag_scheme) if config.head == "crf" else None

    @classmethod
    def from_pretrained(
        cls,
        encoder: EncoderConfig,
        model: ModelConfig,
        *,
        source: Path | None = None,
    ) -> TokenTagger:
        """Загружает закреплённую ревизию pretrained encoder-а."""
        reference: str | Path = source or encoder.name
        loading = {
            "trust_remote_code": encoder.trust_remote_code,
            "use_safetensors": encoder.use_safetensors,
            "dtype": torch.float32,
        }
        if source is None:
            loading["revision"] = encoder.revision
        else:
            loading["local_files_only"] = True
        with _safetensors_conversion_disabled(encoder.use_safetensors is False):
            backbone = AutoModel.from_pretrained(reference, **loading)
        return cls(backbone, model)

    def enable_gradient_checkpointing(self) -> None:
        """Включает экономию VRAM, если encoder её поддерживает."""
        method = getattr(self.encoder, "gradient_checkpointing_enable", None)
        if method is None:
            raise ValueError("Encoder не поддерживает gradient checkpointing")
        method()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> TaggerOutput:
        """Считает emissions и loss для одного batch."""
        if token_type_ids is None:
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            encoded = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        logits = self.classifier(self.dropout(encoded.last_hidden_state))
        if labels is None:
            return TaggerOutput(logits=logits, loss=None)
        if self.crf is not None:
            loss = self.crf.neg_log_likelihood(logits, labels)
        else:
            loss = functional.cross_entropy(
                logits.reshape(-1, len(self.tags)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return TaggerOutput(logits=logits, loss=loss)

    def decode(self, emissions: torch.Tensor) -> tuple[int, ...]:
        """Декодирует одну полную последовательность emissions."""
        if emissions.ndim != 2 or emissions.shape[1] != len(self.tags):
            raise ValueError("Ожидается матрица sequence x tags")
        if emissions.shape[0] == 0:
            return ()
        if self.crf is not None:
            return self.crf.decode_one(emissions.to(self.crf.transitions.device))
        if self.model_config.decoder == "constrained":
            return constrained_viterbi(
                emissions.detach().float().cpu().tolist(),
                self.tags,
                self.model_config.tag_scheme,
            )
        greedy = emissions.argmax(dim=-1).detach().cpu().tolist()
        return tuple(self._repair_bio(greedy))

    def _repair_bio(self, path: Sequence[int]) -> list[int]:
        """Повторяет official greedy-repair для честного BIO baseline."""
        if self.model_config.tag_scheme != "bio":
            raise ValueError("Greedy repair поддерживает только BIO")
        repaired: list[int] = []
        previous_tag = "O"
        for index in path:
            tag = self.tags[index]
            prefix, label = split_tag(tag)
            previous_prefix, previous_label = split_tag(previous_tag)
            if prefix == "I" and not (previous_prefix in {"B", "I"} and previous_label == label):
                tag = f"B-{label}"
                index = self.tags.index(tag)
            repaired.append(index)
            previous_tag = tag
        return repaired
