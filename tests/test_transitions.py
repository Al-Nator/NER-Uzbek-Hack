"""Тесты constrained decoding."""

import pytest

from uzner.data.tagging import build_tag_vocabulary
from uzner.models.transitions import (
    allowed_end,
    allowed_start,
    allowed_transition,
    constrained_viterbi,
    split_tag,
)


def test_bio_transition_rules() -> None:
    """BIO запрещает старт с I и смену класса внутри I."""
    assert not allowed_start("I-NAME", "bio")
    assert allowed_start("B-NAME", "bio")
    assert allowed_end("I-NAME", "bio")
    assert allowed_transition("B-NAME", "I-NAME", "bio")
    assert not allowed_transition("B-ORG", "I-NAME", "bio")
    assert allowed_transition("I-NAME", "B-GEO", "bio")


def test_bioes_transition_rules() -> None:
    """BIOES требует закрывать многотокенную сущность тегом E."""
    assert not allowed_end("B-NAME", "bioes")
    assert allowed_end("E-NAME", "bioes")
    assert allowed_transition("B-NAME", "E-NAME", "bioes")
    assert not allowed_transition("B-NAME", "O", "bioes")
    assert allowed_transition("S-NAME", "B-GEO", "bioes")


def test_viterbi_replaces_locally_best_invalid_path() -> None:
    """Viterbi выбирает валидную последовательность вместо O -> I."""
    tags = build_tag_vocabulary("bio")
    outside = tags.index("O")
    begin = tags.index("B-NAME")
    inside = tags.index("I-NAME")
    emissions = [[0.0] * len(tags) for _ in range(2)]
    emissions[0][outside] = 5.0
    emissions[0][begin] = 4.0
    emissions[1][inside] = 5.0

    path = constrained_viterbi(emissions, tags, "bio")

    assert tuple(tags[index] for index in path) == ("B-NAME", "I-NAME")


def test_viterbi_supports_transition_scores_and_empty_input() -> None:
    """Decoder учитывает обучаемые переходы и пустую последовательность."""
    tags = ("O", "B-NAME")
    transitions = ((0.0, 10.0), (0.0, 0.0))

    assert constrained_viterbi([], tags, "bio") == ()
    assert constrained_viterbi(((1.0, 0.0), (1.0, 0.0)), tags, "bio", transitions) == (0, 1)


def test_transition_validation() -> None:
    """Ошибочные теги и размерности дают явную ошибку."""
    with pytest.raises(ValueError, match="Некорректный тег"):
        split_tag("NAME")
    with pytest.raises(ValueError, match="Размерность"):
        constrained_viterbi(((1.0,),), ("O", "B-NAME"), "bio")
    with pytest.raises(ValueError, match="Матрица"):
        constrained_viterbi(((1.0, 0.0),), ("O", "B-NAME"), "bio", ((0.0,),))
