"""Простые правила границ, повторов и train-словаря без доступа к dev gold."""

from collections import defaultdict
from dataclasses import replace

from uzner.domain import Document, Entity, Prediction
from uzner.evaluation.repeated_mentions import RepeatConfig, propagate_repeats
from uzner.posthoc.config import Variant
from uzner.posthoc.lexicon import APOSTROPHES, Lexicon, LexiconEntry, scan_keys, surface_key
from uzner.training.span_prediction import select_flat_entities


def overlaps(left: Entity, right: Entity) -> bool:
    """Проверяет пересечение полуоткрытых интервалов, разрешая соседние spans."""
    return left.start < right.end and right.start < left.end


def add_safe(base: Prediction, additions: list[Entity]) -> Prediction:
    """Добавляет непересекающиеся кандидаты, никогда не вытесняя исходные сущности."""
    candidates = [e for e in additions if not any(overlaps(e, old) for old in base.entities)]
    return Prediction(base.hash, base.entities + select_flat_entities(candidates))


def qualified(entry: LexiconEntry | None, variant: Variant) -> bool:
    """Проверяет независимые документы, однозначность и частоту разметки формы."""
    return bool(
        entry
        and entry.support >= variant.support
        and entry.purity >= variant.purity
        and entry.propensity >= variant.propensity
        and entry.label in variant.labels
    )


def repeat(document: Document, prediction: Prediction, variant: Variant) -> Prediction:
    """Переносит метку только с исходных упоминаний; рекурсивного расширения нет."""
    if not variant.normalized and variant.min_seeds == 1:
        seeds = Prediction(
            prediction.hash, tuple(e for e in prediction.entities if e.label in variant.labels)
        )
        expanded = propagate_repeats(document, seeds, RepeatConfig(variant.min_length))
        return add_safe(prediction, list(expanded.entities))
    seeds = defaultdict(list)
    for entity in prediction.entities:
        surface = document.text[entity.start : entity.end]
        if len(surface) >= variant.min_length:
            seeds[surface_key(surface, variant.normalized)].append(entity)
    allowed = {
        key: entities[0].label
        for key, entities in seeds.items()
        if len({e.label for e in entities}) == 1
        and len(entities) >= variant.min_seeds
        and entities[0].label in variant.labels
    }
    prefixes = frozenset(key[:n] for key in allowed for n in range(1, len(key) + 1))
    additions = [
        Entity(start, end, allowed[key])
        for key, start, end in scan_keys(document.text, prefixes, variant.normalized, 12)
        if key in allowed
    ]
    return add_safe(prediction, additions)


def word_character(char: str) -> bool:
    """Распознаёт буквы и внутренние знаки слов без изменения Unicode-строки."""
    return char.isalnum() or char in APOSTROPHES + "_-"


def proposed_boundary(text: str, entity: Entity, mode: str) -> Entity | None:
    """Предлагает небольшой сдвиг в пределах текущего слова или внешних кавычек."""
    start, end = entity.start, entity.end
    if mode in {"trim", "conservative"}:
        while start < end and text[start] in ' \t\n«»“”„"([{@#':
            start += 1
        while end > start and text[end - 1] in ' \t\n«»“”„",;:!?)]}':
            # Закрывающая скобка внутри названия сохраняется при парной открывающей.
            if text[end - 1] in ")]}":
                opener = {")": "(", "]": "[", "}": "{"}[text[end - 1]]
                if opener in text[start : end - 1]:
                    break
            end -= 1
    if start == end:
        return None
    if mode in {"word", "conservative", "dictionary"}:
        while start and word_character(text[start - 1]) and word_character(text[start]):
            start -= 1
    if mode in {"right", "word", "conservative", "dictionary"}:
        right = end
        while right < len(text) and word_character(text[right]) and word_character(text[right - 1]):
            right += 1
        suffix = text[end:right]
        if mode != "conservative" or (len(suffix) <= 10 and not any(c.isupper() for c in suffix)):
            end = right
    return replace(entity, start=start, end=end)


def boundaries(
    document: Document, prediction: Prediction, variant: Variant, lexicon: Lexicon
) -> Prediction:
    """Не допускает пересечения новой границы с другой исходной сущностью."""
    result = []
    for entity in prediction.entities:
        candidate = proposed_boundary(document.text, entity, variant.boundary)
        if entity.label not in variant.labels:
            candidate = entity
        if candidate is None:
            continue
        if variant.boundary == "dictionary":
            entry = lexicon.get(document.text[candidate.start : candidate.end])
            if not qualified(entry, variant) or entry.label != entity.label:
                candidate = entity
        if any(overlaps(candidate, old) for old in prediction.entities if old is not entity):
            candidate = entity
        result.append(candidate)
    # Две независимые границы могут расшириться навстречу; при конфликте отменяем обе.
    conflicting = {
        i
        for i, left in enumerate(result)
        for j, right in enumerate(result)
        if i != j and overlaps(left, right)
    }
    if conflicting:
        return prediction
    return Prediction(prediction.hash, tuple(result))


def dictionary_rule(
    document: Document, prediction: Prediction, variant: Variant, lexicon: Lexicon
) -> Prediction:
    """Добавляет или уточняет словарные формы без подмены контекстного декодера."""
    if variant.operation == "lex_add":
        additions = [
            Entity(m.start, m.end, m.entry.label, m.entry.propensity)
            for m in lexicon.matches(document.text)
            if qualified(m.entry, variant) and m.end - m.start >= variant.min_length
        ]
        return add_safe(prediction, additions)
    entities = []
    for entity in prediction.entities:
        surface = document.text[entity.start : entity.end]
        entry = lexicon.get(surface)
        if variant.operation == "short" and len(surface) <= variant.min_length and entry is None:
            continue
        if variant.operation == "lex_relabel" and qualified(entry, variant):
            entity = replace(entity, label=entry.label)
        entities.append(entity)
    return Prediction(prediction.hash, tuple(entities))
