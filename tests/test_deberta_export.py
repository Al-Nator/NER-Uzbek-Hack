"""Контракт экспортной специализации: только равные query/key, без изменения позиций."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.deberta_v2 import modeling_deberta_v2 as deberta

from uzner.serving.deberta_export import export_self_attention, self_attention_rpos


@pytest.mark.parametrize("length", [2, 7, 32, 128, 257, 512])
def test_relative_positions_unchanged(length):
    """Сравнивает штатный helper и экспортную ветку на крайних и промежуточных длинах."""
    query = torch.zeros((2, length, 4))
    position = deberta.build_relative_position(query, query, 256, 512).unsqueeze(0)
    assert torch.equal(
        deberta.build_rpos(query, query, position, 256, 512),
        self_attention_rpos(query, query, position, 256, 512),
    )


def test_cross_attention_rejected():
    """Не позволяет применять специализацию к разным длинам query/key."""
    with pytest.raises(ValueError, match="self-attention"):
        self_attention_rpos(torch.zeros(1, 2, 4), torch.zeros(1, 3, 4), torch.zeros(2, 3), 256, 512)


def test_context_restores_original_on_failure():
    """Не оставляет изменённый Transformers helper после ошибки экспорта."""
    original = deberta.build_rpos
    encoder = SimpleNamespace(config=SimpleNamespace(model_type="deberta-v2", is_decoder=False))
    with pytest.raises(RuntimeError), export_self_attention(encoder):
        assert deberta.build_rpos is self_attention_rpos
        raise RuntimeError("export failed")
    assert deberta.build_rpos is original


def test_other_encoder_not_modified():
    """Не меняет путь экспорта XLM-R/BGE."""
    original = deberta.build_rpos
    with export_self_attention(SimpleNamespace(config=SimpleNamespace(model_type="xlm-roberta"))):
        assert deberta.build_rpos is original


def test_decoder_rejected():
    """Не специализирует потенциальный cross-attention decoder."""
    encoder = SimpleNamespace(config=SimpleNamespace(model_type="deberta-v2", is_decoder=True))
    with pytest.raises(ValueError, match="decoder"), export_self_attention(encoder):
        pass
