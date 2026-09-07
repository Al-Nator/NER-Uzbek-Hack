"""Добавление точных повторных упоминаний без изменения исходных spans."""

from dataclasses import dataclass

from uzner.domain import Document, Entity, Prediction


@dataclass(frozen=True)
class RepeatConfig:
    """Фиксированное правило проверенной абляции повторов."""

    min_length: int = 4
    word_characters: str = "'’ʻʼ‘`-"


def propagate_repeats(
    document: Document, prediction: Prediction, config: RepeatConfig | None = None
) -> Prediction:
    """Копирует метку на точные непересекающиеся вхождения с границами слова."""
    config = config if config is not None else RepeatConfig()
    if document.hash != prediction.hash:
        raise ValueError("hash документа и предсказания различаются")
    if config.min_length < 1:
        raise ValueError("Минимальная длина должна быть положительной")
    Document(document.hash, document.text, prediction.entities)
    text = document.text
    entities = list(prediction.entities)
    for seed in prediction.entities:
        surface = text[seed.start : seed.end]
        if len(surface) < config.min_length:
            continue
        search_from = 0
        while (start := text.find(surface, search_from)) >= 0:
            end = start + len(surface)
            search_from = start + 1
            left_inside = start > 0 and all(
                c.isalnum() or c in config.word_characters for c in text[start - 1 : start + 1]
            )
            right_inside = end < len(text) and all(
                c.isalnum() or c in config.word_characters for c in text[end - 1 : end + 1]
            )
            if left_inside or right_inside:
                continue
            if any(start < old.end and old.start < end for old in entities):
                continue
            entities.append(Entity(start=start, end=end, label=seed.label, score=seed.score))
    return Prediction(document.hash, tuple(entities))
