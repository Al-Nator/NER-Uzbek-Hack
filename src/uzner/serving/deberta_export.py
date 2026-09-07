"""Экспорт self-attention DeBERTa без недостижимой ветки разных длин query/key."""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import torch


def self_attention_rpos(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    relative_pos: torch.Tensor,
    position_buckets: int,
    max_relative_positions: int,
) -> torch.Tensor:
    """Возвращает исходные позиции; cross-attention этим экспортом не поддерживается."""
    if not torch.onnx.is_in_onnx_export() and query_layer.shape[-2] != key_layer.shape[-2]:
        raise ValueError("Экспорт DeBERTa разрешён только для self-attention")
    return relative_pos


@contextmanager
def export_self_attention(encoder: torch.nn.Module) -> Iterator[None]:
    """Локально убирает ONNX If с разными рангами; исходный helper восстанавливается."""
    if getattr(encoder.config, "model_type", "") != "deberta-v2":
        yield
        return
    if getattr(encoder.config, "is_decoder", False):
        raise ValueError("Экспорт decoder/cross-attention DeBERTa не поддерживается")
    from transformers.models.deberta_v2 import modeling_deberta_v2

    # AutoModel encoder получает единую последовательность, поэтому длины query/key равны.
    # Используется только отдельным процессом экспорта, не inference-сервисом.
    with patch.object(modeling_deberta_v2, "build_rpos", self_attention_rpos):
        yield
