"""Применение одной детерминированной posthoc-гипотезы к готовым предсказаниям."""

from dataclasses import dataclass, replace

from uzner.domain import Document, Prediction
from uzner.evaluation.span_vote import majority_vote
from uzner.posthoc.config import Variant
from uzner.posthoc.lexicon import Lexicon
from uzner.posthoc.rules import boundaries, dictionary_rule, repeat
from uzner.posthoc.selection import confirmed, decode_candidates, vote_four


@dataclass(frozen=True)
class Inputs:
    """Входы декодера без золота; словари заранее построены только по train."""

    documents: tuple[Document, ...]
    sources: dict[str, tuple[Prediction, ...]]
    candidates: tuple[Prediction, ...]
    exact_lexicon: Lexicon
    normalized_lexicon: Lexicon
    candidate_floor: float


def apply_variant(inputs: Inputs, variant: Variant) -> tuple[Prediction, ...]:
    """Отделяет декодирование от evaluator и запрещает передачу gold в правила."""
    if any(document.entities for document in inputs.documents):
        raise ValueError("Posthoc decoder не должен получать gold-сущности")
    base = inputs.sources[variant.base]
    components = tuple(inputs.sources[key] for key in ("s33", "s21", "s31"))
    lexicon = inputs.normalized_lexicon if variant.normalized else inputs.exact_lexicon
    if variant.operation in {"gp", "gp_vote"}:
        limits = (variant.threshold, variant.org_threshold, variant.name_threshold)
        if any(value is not None and value < inputs.candidate_floor for value in limits):
            raise ValueError("Порог ниже floor кэша: часть нужных кандидатов не сохранена")
        decoded = tuple(decode_candidates(p.hash, p.entities, variant) for p in inputs.candidates)
        base = (
            majority_vote(inputs.documents, (decoded, components[1], components[2]))
            if variant.operation == "gp_vote"
            else decoded
        )
    result = []
    for index, document in enumerate(inputs.documents):
        prediction = base[index]
        if prediction.hash != document.hash:
            raise ValueError("Нарушен порядок документов и предсказаний")
        if variant.operation == "repeat":
            prediction = repeat(document, prediction, variant)
        elif variant.operation == "boundary":
            prediction = boundaries(document, prediction, variant, lexicon)
        elif variant.operation in {"lex_add", "lex_relabel", "short"}:
            prediction = dictionary_rule(document, prediction, variant, lexicon)
        elif variant.operation in {"confirm", "lex_confirm"}:
            prediction = confirmed(
                document.text,
                prediction,
                tuple(p[index] for p in components),
                inputs.sources[variant.confirmer][index],
                variant,
                lexicon,
            )
        elif variant.operation == "vote":
            prediction = vote_four(
                prediction,
                tuple(p[index] for p in components) + (inputs.sources[variant.confirmer][index],),
            )
        if variant.boundary_after:
            prediction = boundaries(document, prediction, variant, lexicon)
        if variant.repeat_after:
            prediction = repeat(
                document,
                prediction,
                replace(
                    variant,
                    normalized=False,
                    min_seeds=1,
                    min_length=4,
                    labels=("ORG", "NAME", "GEO"),
                ),
            )
        Document(document.hash, document.text, prediction.entities)
        result.append(prediction)
    return tuple(result)
