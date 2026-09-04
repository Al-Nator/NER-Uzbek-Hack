"""Тесты окон, token tagger, CRF и объединения logits."""

import os

import pytest
import torch
from torch import nn

from tests.helpers import FakeTokenizer, TinyEncoder, sample_documents
from uzner.config import EncoderConfig, ModelConfig, TokenizationConfig
from uzner.data.tagging import build_tag_vocabulary
from uzner.data.windows import (
    IGNORE_LABEL,
    WindowCollator,
    WindowDataset,
    WindowFeature,
    align_window_labels,
    build_window_features,
    canonicalize_offsets,
    collect_chunk_edges,
)
from uzner.domain import Document, Entity
from uzner.models.crf import INVALID_SCORE, LinearChainCrf
from uzner.models.token_tagger import TokenTagger
from uzner.training.prediction import WindowLogits, aggregate_window_logits, decode_documents


def _model(scheme: str = "bio", head: str = "softmax", decoder: str = "greedy") -> TokenTagger:
    """Создаёт маленький tagger заданного типа."""
    return TokenTagger(
        TinyEncoder(),
        ModelConfig("token_tagging", scheme, head, decoder, dropout=0.0),
    )


def test_window_alignment_masks_partial_entity() -> None:
    """Сущность на краю окна не создаёт ложную метку."""
    offsets = ((0, 0), (4, 7), (8, 12), (0, 0))
    labels = align_window_labels(
        offsets,
        (Entity(label="NAME", start=2, end=7), Entity(label="GEO", start=8, end=12)),
        "bio",
    )

    tags = build_tag_vocabulary("bio")
    assert labels[0] == labels[-1] == IGNORE_LABEL
    assert labels[1] == IGNORE_LABEL
    assert labels[2] == tags.index("B-GEO")
    with pytest.raises(ValueError, match="не содержит"):
        align_window_labels(((0, 0),), (), "bio")


@pytest.mark.parametrize("scheme", ("bio", "bioes"))
def test_window_alignment_masks_entities_merged_into_one_token(scheme: str) -> None:
    """Две gold-сущности в одном токене маскируются без ложной разметки."""
    offsets = ((0, 0), (0, 3), (3, 8), (0, 0))
    entities = (
        Entity(label="NAME", start=0, end=4),
        Entity(label="ORG", start=4, end=8),
    )

    labels = align_window_labels(offsets, entities, scheme)

    assert labels == (IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL)


def test_offsets_drop_only_external_tokenizer_whitespace() -> None:
    """Пробел перед словом не сдвигает exact span у byte-level tokenizer-а."""
    text = "Ali \n Банк  ichida"

    offsets = canonicalize_offsets(text, ((0, 0), (3, 10), (10, 12), (10, 17), (0, 0)))

    assert offsets == ((0, 0), (6, 10), (12, 12), (12, 17), (0, 0))


def test_window_build_canonicalizes_prefixed_space_before_alignment() -> None:
    """BIO-разметка принимает соседние сущности при prefixed-space offsets."""

    class PrefixSpaceTokenizer:
        """Возвращает характерные для SentencePiece offsets с левым пробелом."""

        def __call__(self, _text: str, **_kwargs: object) -> dict[str, list[object]]:
            """Имитирует одно окно с двумя соседними сущностями."""
            return {
                "input_ids": [1, 2, 3, 4],
                "attention_mask": [1, 1, 1, 1],
                "offset_mapping": [(0, 0), (0, 3), (3, 7), (0, 0)],
            }

    document = Document(
        hash="space",
        text="Ali Bob",
        source="test",
        entities=(
            Entity(label="NAME", start=0, end=3),
            Entity(label="NAME", start=4, end=7),
        ),
    )

    feature = build_window_features(
        (document,),
        PrefixSpaceTokenizer(),
        TokenizationConfig(max_length=8, stride=2),
        "bio",
        with_labels=True,
    )[0]

    name_tag = build_tag_vocabulary("bio").index("B-NAME")
    assert feature.offsets == ((0, 0), (0, 3), (4, 7), (0, 0))
    assert feature.labels == (IGNORE_LABEL, name_tag, name_tag, IGNORE_LABEL)


def test_window_build_dataset_collator_and_edges() -> None:
    """Окна сохраняют offsets, метки и дополнение batch."""
    documents = sample_documents()
    features = build_window_features(
        documents,
        FakeTokenizer(),
        TokenizationConfig(max_length=8, stride=2),
        "bioes",
        with_labels=True,
    )
    dataset = WindowDataset(features)
    batch = WindowCollator(0)((dataset[0], dataset[-1]))

    assert len(dataset) == len(features) >= 2
    assert batch.input_ids.shape[0] == 2
    assert batch.labels is not None
    assert batch.token_type_ids is not None
    assert collect_chunk_edges(features)[0]
    batch.to(torch.device("cpu"))

    unlabeled = WindowFeature(0, 0, (1,), (1,), None, ((0, 1),), None)
    labeled = WindowFeature(0, 1, (1,), (1,), None, ((0, 1),), (0,))
    with pytest.raises(ValueError, match="не должен"):
        WindowDataset(())
    with pytest.raises(ValueError, match="Batch"):
        WindowCollator(0)(())
    with pytest.raises(ValueError, match="смешивать"):
        WindowCollator(0)((unlabeled, labeled))


def test_crf_loss_gradients_and_valid_decode() -> None:
    """CRF считает NLL, градиенты и не нарушает BIO."""
    tags = build_tag_vocabulary("bio")
    crf = LinearChainCrf(tags, "bio")
    logits = torch.randn(2, 4, len(tags), requires_grad=True)
    labels = torch.tensor([[0, 3, 4, -100], [1, 2, 0, -100]])

    loss = crf.neg_log_likelihood(logits, labels)
    loss.backward()
    path = crf.decode_one(torch.randn(4, len(tags)))

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert tags[path[0]] != "I-ORG"
    assert float(crf._masked_start()[tags.index("I-NAME")]) == INVALID_SCORE

    with pytest.raises(ValueError, match="непустой"):
        LinearChainCrf((), "bio")
    with pytest.raises(ValueError, match="размерности"):
        crf.neg_log_likelihood(torch.zeros(2, 3), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="обучаемых"):
        crf.neg_log_likelihood(torch.zeros(1, 2, len(tags)), torch.full((1, 2), -100))
    with pytest.raises(ValueError, match="nonempty"):
        crf.decode_one(torch.zeros(0, len(tags)))


def test_token_tagger_softmax_crf_and_decoders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tagger покрывает forward и три варианта decoding."""
    softmax = _model()
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    labels = torch.tensor([[-100, 3, 4]])
    plain = softmax(inputs, mask)
    trained = softmax(inputs, mask, torch.zeros_like(inputs), labels)

    assert plain.loss is None
    assert trained.loss is not None
    assert plain.logits.shape == (1, 3, 7)
    inside = softmax.tags.index("I-NAME")
    emissions = torch.zeros(2, len(softmax.tags))
    emissions[:, inside] = 5
    repaired = softmax.decode(emissions)
    assert softmax.tags[repaired[0]] == "B-NAME"
    assert softmax.decode(torch.zeros(0, len(softmax.tags))) == ()

    constrained = _model(decoder="constrained")
    assert len(constrained.decode(torch.randn(2, len(constrained.tags)))) == 2
    crf = _model(head="crf", decoder="crf")
    assert crf(inputs, mask, labels=labels).loss is not None
    assert len(crf.decode(torch.randn(2, len(crf.tags)))) == 2

    softmax.enable_gradient_checkpointing()
    assert softmax.encoder.checkpointing_enabled
    received: dict[str, object] = {}
    monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION", raising=False)

    def fake_from_pretrained(*args: object, **kwargs: object) -> TinyEncoder:
        """Сохраняет параметры загрузки и возвращает маленький encoder."""
        received.update(kwargs)
        return TinyEncoder()

    monkeypatch.setattr(
        "uzner.models.token_tagger.AutoModel.from_pretrained",
        fake_from_pretrained,
    )
    loaded = TokenTagger.from_pretrained(
        EncoderConfig("local", "revision", use_safetensors=False),
        softmax.model_config,
    )
    assert isinstance(loaded.encoder, TinyEncoder)
    assert received["use_safetensors"] is False
    assert received["dtype"] is torch.float32
    assert "DISABLE_SAFETENSORS_CONVERSION" not in os.environ

    with pytest.raises(ValueError, match="hidden_size"):
        TokenTagger(nn.Identity(), softmax.model_config)
    with pytest.raises(ValueError, match="матрица"):
        softmax.decode(torch.zeros(2, 2))
    softmax.model_config = ModelConfig("token_tagging", "bioes", "softmax", "constrained")
    with pytest.raises(ValueError, match="Greedy"):
        softmax._repair_bio((0,))


def test_window_aggregation_and_document_decode() -> None:
    """Перекрытия усредняются, а special offsets игнорируются."""
    model = _model()
    tag_count = len(model.tags)
    first = torch.zeros(3, tag_count)
    second = torch.zeros(3, tag_count)
    first[1, model.tags.index("B-NAME")] = 4
    second[1, model.tags.index("B-NAME")] = 4
    windows = (
        WindowLogits(0, ((0, 0), (0, 3), (4, 8)), first),
        WindowLogits(0, ((0, 3), (4, 8), (0, 0)), second),
    )
    emissions = aggregate_window_logits(windows, 1, average_probabilities=True)
    document = Document(hash="one", text="Ali joy", source="test")
    predictions = decode_documents((document,), emissions, model)

    assert emissions[0].offsets == ((0, 3), (4, 8))
    assert predictions[0].hash == "one"
    raw = aggregate_window_logits(windows, 1, average_probabilities=False)
    assert raw[0].values.shape == (2, tag_count)

    with pytest.raises(ValueError, match="положительным"):
        aggregate_window_logits(windows, 0, average_probabilities=True)
    with pytest.raises(ValueError, match="выходит"):
        aggregate_window_logits(
            (WindowLogits(2, ((0, 1),), torch.zeros(1, 2)),),
            1,
            average_probabilities=False,
        )
    with pytest.raises(ValueError, match="offsets"):
        aggregate_window_logits(
            (WindowLogits(0, ((0, 1),), torch.zeros(2, 2)),),
            1,
            average_probabilities=False,
        )
    with pytest.raises(ValueError, match="не вернул"):
        aggregate_window_logits(
            (WindowLogits(0, ((0, 0),), torch.zeros(1, 2)),),
            1,
            average_probabilities=False,
        )
    with pytest.raises(ValueError, match="совпадать"):
        decode_documents((document,), (), model)
