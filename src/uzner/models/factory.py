"""Выбор модели без архитектурных веток в training/checkpoint-контуре."""

from uzner.config import ModelConfig
from uzner.models.span_tagger import SpanTagger
from uzner.models.token_tagger import TokenTagger


def model_class(config: ModelConfig) -> type[TokenTagger]:
    """Возвращает реализацию token или span head."""
    return SpanTagger if config.architecture == "span" else TokenTagger
