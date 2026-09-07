"""Выбор плоских spans из вероятностей и подтверждений независимых моделей."""

from bisect import bisect_right
from collections import Counter

from uzner.domain import Entity, Prediction
from uzner.posthoc.config import Variant
from uzner.posthoc.lexicon import Lexicon
from uzner.posthoc.rules import add_safe, qualified
from uzner.training.span_prediction import select_flat_entities


def entity_key(entity: Entity) -> tuple[int, int, str]:
    """Возвращает ключ точного совпадения без несущественного поля score."""
    return entity.start, entity.end, entity.label


def threshold_for(label: str, variant: Variant) -> float:
    """Разрешает явно заданный порог класса или общий порог GP."""
    if label == "ORG" and variant.org_threshold is not None:
        return variant.org_threshold
    if label == "NAME" and variant.name_threshold is not None:
        return variant.name_threshold
    return variant.threshold


def optimal_flat(candidates: list[Entity], variant: Variant) -> tuple[Entity, ...]:
    """Максимизирует сумму превышений порога на непересекающихся интервалах."""
    ordered = sorted(candidates, key=lambda e: (e.end, e.start, e.label))
    ends = [e.end for e in ordered]
    best, previous, taken = [0.0], [], []
    for index, entity in enumerate(ordered):
        before = bisect_right(ends, entity.start, 0, index)
        weight = (entity.score or 0) - threshold_for(entity.label, variant)
        include = best[before] + weight
        take = include > best[-1]
        best.append(include if take else best[-1])
        previous.append(before)
        taken.append(take)
    selected, cursor = [], len(ordered)
    while cursor:
        if taken[cursor - 1]:
            selected.append(ordered[cursor - 1])
            cursor = previous[cursor - 1]
        else:
            cursor -= 1
    return tuple(sorted(selected, key=lambda e: (e.start, e.end, e.label)))


def decode_candidates(
    hash_value: str, candidates: tuple[Entity, ...], variant: Variant
) -> Prediction:
    """Применяет порог к pre-NMS кандидатам, а не к уже усечённому предсказанию."""
    if any(e.score is None for e in candidates):
        raise ValueError("Кандидаты GP должны содержать score")
    selected = [e for e in candidates if e.score > threshold_for(e.label, variant)]
    entities = (
        optimal_flat(selected, variant)
        if variant.selection == "optimal"
        else select_flat_entities(selected)
    )
    return Prediction(hash_value, entities)


def confirmed(
    text: str,
    base: Prediction,
    components: tuple[Prediction, ...],
    confirmer: Prediction,
    variant: Variant,
    lexicon: Lexicon,
) -> Prediction:
    """Добавляет singleton только с новым модельным или train-словарным подтверждением."""
    votes = Counter(entity_key(e) for p in components for e in p.entities)
    support = {entity_key(e) for e in confirmer.entities}
    additions = []
    for (start, end, label), count in votes.items():
        if label not in variant.labels or count != 1:
            continue
        accepted = (start, end, label) in support
        if variant.operation == "lex_confirm":
            entry = lexicon.get(text[start:end])
            accepted = qualified(entry, variant) and entry.label == label
        if accepted:
            additions.append(Entity(start, end, label, count / 3))
    return add_safe(base, additions)


def vote_four(base: Prediction, sources: tuple[Prediction, ...]) -> Prediction:
    """Добавляет новые совпадения 2/4, сохраняя все spans исходного ансамбля."""
    if len(sources) != 4:
        raise ValueError("Ожидаются четыре разных источника")
    votes = Counter(entity_key(e) for p in sources for e in p.entities)
    return add_safe(
        base,
        [Entity(a, b, label, count / 4) for (a, b, label), count in votes.items() if count >= 2],
    )
