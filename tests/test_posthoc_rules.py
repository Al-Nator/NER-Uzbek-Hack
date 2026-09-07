"""Регрессии повторов, границ, словарей и плоского выбора интервалов."""

from dataclasses import replace

import pytest

from uzner.domain import Document, Entity, Prediction
from uzner.posthoc.config import Variant
from uzner.posthoc.lexicon import build_lexicon
from uzner.posthoc.rules import add_safe, boundaries, dictionary_rule, repeat
from uzner.posthoc.selection import confirmed, decode_candidates, optimal_flat, vote_four


@pytest.fixture
def lexicon():
    """Создаёт небольшой train-only словарь для изолированных тестов."""
    return build_lexicon(
        tuple(
            Document(str(i), "Ali Toshkent", (Entity(0, 3, "NAME"), Entity(4, 12, "GEO")))
            for i in range(5)
        )
    )


def test_add_safe_never_overwrites_base():
    """Высокий score нового кандидата не вытесняет исходное решение."""
    old = Prediction("d", (Entity(0, 5, "ORG"),))
    assert add_safe(old, [Entity(0, 6, "NAME", 1.0), Entity(5, 8, "GEO")]).entities == (
        Entity(0, 5, "ORG"),
        Entity(5, 8, "GEO"),
    )


def test_repeat_normalized_and_no_recursive_vote():
    """Поиск нечувствителен к регистру только явно и требует заданного числа seed."""
    doc = Document("d", "Ali ALI Aliyev")
    pred = Prediction("d", (Entity(0, 3, "NAME"),))
    variant = Variant("repeat", "repeat", min_length=2, normalized=True)
    assert len(repeat(doc, pred, variant).entities) == 2
    assert repeat(doc, pred, replace(variant, min_seeds=2)) == pred
    assert repeat(doc, pred, replace(variant, labels=("GEO",))) == pred


def test_conflicting_repeat_labels_are_not_propagated():
    """Противоречивые метки одной нормализованной формы не размножаются."""
    doc = Document("d", "Ali ALI ali")
    pred = Prediction("d", (Entity(0, 3, "NAME"), Entity(4, 7, "GEO")))
    assert repeat(doc, pred, Variant("r", "repeat", normalized=True, min_length=2)) == pred


def test_exact_label_subset_does_not_normalize_spaces():
    """Ограничение классов не должно незаметно заменять точный поиск токенным."""
    doc = Document("d", "New York New\nYork New York")
    pred = Prediction("d", (Entity(0, 8, "ORG"),))
    result = repeat(doc, pred, Variant("r", "repeat", labels=("ORG",)))
    assert [(e.start, e.end) for e in result.entities] == [(0, 8), (18, 26)]


@pytest.mark.parametrize(
    "mode,expected",
    [("right", (2, 7)), ("word", (0, 7)), ("conservative", (0, 7)), ("trim", (2, 3))],
)
def test_boundary_modes(mode, expected, lexicon):
    """Сдвиги остаются в пределах слова и сохраняют character offsets."""
    doc = Document("d", "Alining!")
    pred = Prediction("d", (Entity(2, 3, "NAME"),))
    out = boundaries(doc, pred, Variant("b", "boundary", boundary=mode), lexicon)
    assert (out.entities[0].start, out.entities[0].end) == expected


def test_boundary_trim_and_collision(lexicon):
    """Удаление внешних кавычек не склеивает соседние сущности."""
    doc = Document("d", '"Ali"')
    out = boundaries(
        doc,
        Prediction("d", (Entity(0, 5, "NAME"),)),
        Variant("b", "boundary", boundary="trim"),
        lexicon,
    )
    assert out.entities == (Entity(1, 4, "NAME"),)
    doc = Document("d", "AliX")
    pred = Prediction("d", (Entity(0, 3, "NAME"), Entity(3, 4, "ORG")))
    assert boundaries(doc, pred, Variant("b", "boundary", boundary="right"), lexicon) == pred
    punctuation = boundaries(
        Document("p", '""'),
        Prediction("p", (Entity(0, 2, "ORG"),)),
        Variant("b", "boundary", boundary="trim"),
        lexicon,
    )
    assert not punctuation.entities


def test_dictionary_boundary_and_relabel(lexicon):
    """Словарное уточнение требует полной формы с достаточной статистикой train."""
    doc = Document("d", "Ali")
    pred = Prediction("d", (Entity(0, 2, "NAME"),))
    assert boundaries(
        doc, pred, Variant("b", "boundary", boundary="dictionary"), lexicon
    ).entities == (Entity(0, 3, "NAME"),)
    wrong = Prediction("d", (Entity(0, 3, "ORG"),))
    assert (
        dictionary_rule(doc, wrong, Variant("l", "lex_relabel"), lexicon).entities[0].label
        == "NAME"
    )
    added = dictionary_rule(doc, Prediction("d"), Variant("a", "lex_add", min_length=2), lexicon)
    assert added.entities[0].start == 0
    assert not dictionary_rule(
        Document("d", "Xi"),
        Prediction("d", (Entity(0, 2, "NAME"),)),
        Variant("s", "short", min_length=2),
        lexicon,
    ).entities


def test_optimal_vs_greedy_and_threshold():
    """Два непересекающихся кандидата могут иметь большую суммарную полезность."""
    candidates = (Entity(0, 6, "ORG", 0.9), Entity(0, 3, "NAME", 0.8), Entity(3, 6, "NAME", 0.8))
    variant = Variant("gp", "gp", threshold=0.2)
    assert len(decode_candidates("d", candidates, variant).entities) == 1
    assert len(optimal_flat(list(candidates), variant)) == 2
    assert not decode_candidates("d", candidates, replace(variant, threshold=1.0)).entities
    assert (
        len(
            decode_candidates(
                "d", candidates, replace(variant, threshold=0.95, name_threshold=0.2)
            ).entities
        )
        == 2
    )
    with pytest.raises(ValueError, match="score"):
        decode_candidates("d", (Entity(0, 3, "NAME"),), variant)


def test_confirmation_not_union(lexicon):
    """Неподтверждённый singleton не добавляется, даже если кандидат существует."""
    entity = Entity(0, 3, "NAME")
    empty, hit = Prediction("d"), Prediction("d", (entity,))
    components = (hit, empty, empty)
    variant = Variant("c", "confirm")
    assert not confirmed("Ali", empty, components, empty, variant, lexicon).entities
    assert len(confirmed("Ali", empty, components, hit, variant, lexicon).entities) == 1
    assert (
        len(
            confirmed(
                "Ali", empty, components, empty, replace(variant, operation="lex_confirm"), lexicon
            ).entities
        )
        == 1
    )
    assert len(vote_four(empty, (*components, hit)).entities) == 1
    with pytest.raises(ValueError):
        vote_four(empty, components)
