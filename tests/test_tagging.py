"""Тесты BIO/BIOES и точных границ."""

import pytest

from uzner.data.tagging import (
    build_tag_vocabulary,
    decode_tags,
    encode_tags,
    is_exactly_representable,
    validate_scheme,
)
from uzner.domain import Entity

OFFSETS = ((0, 0), (0, 3), (4, 12), (13, 20), (0, 0))
ENTITIES = (
    Entity(label="NAME", start=0, end=3),
    Entity(label="GEO", start=4, end=20),
)


@pytest.mark.parametrize("scheme", ["bio", "bioes"])
def test_tagging_roundtrip(scheme: str) -> None:
    """Валидные spans проходят точный roundtrip через обе схемы."""
    tags = encode_tags(OFFSETS, ENTITIES, validate_scheme(scheme))

    assert decode_tags(OFFSETS, tags, validate_scheme(scheme)) == ENTITIES


def test_expected_bio_and_bioes_tags() -> None:
    """Одно- и многотокенные сущности получают правильные префиксы."""
    assert encode_tags(OFFSETS, ENTITIES, "bio") == (
        None,
        "B-NAME",
        "B-GEO",
        "I-GEO",
        None,
    )
    assert encode_tags(OFFSETS, ENTITIES, "bioes") == (
        None,
        "S-NAME",
        "B-GEO",
        "E-GEO",
        None,
    )


def test_vocabulary_and_representability() -> None:
    """Словарь детерминирован, а внутренние границы subword видны аудиту."""
    assert len(build_tag_vocabulary("bio")) == 7
    assert len(build_tag_vocabulary("bioes")) == 13
    assert is_exactly_representable(OFFSETS, ENTITIES[0])
    assert not is_exactly_representable(OFFSETS, Entity(label="NAME", start=1, end=3))
    assert not is_exactly_representable(OFFSETS, Entity(label="ORG", start=30, end=31))


def test_entity_outside_window_is_ignored() -> None:
    """Сущность вне текущего окна не создаёт ложных тегов."""
    tags = encode_tags(
        ((0, 0), (0, 3), (0, 0)),
        (Entity(label="NAME", start=5, end=8),),
        "bio",
    )

    assert tags == (None, "O", None)


def test_invalid_scheme_overlap_and_sequences_are_rejected() -> None:
    """Некорректные схемы, пересечения и переходы не исправляются молча."""
    with pytest.raises(ValueError, match="bioes"):
        validate_scheme("bilou")
    with pytest.raises(ValueError, match="один токен"):
        encode_tags(
            ((0, 3),),
            (
                Entity(label="NAME", start=0, end=3),
                Entity(label="ORG", start=0, end=3),
            ),
            "bio",
        )
    with pytest.raises(ValueError, match="BIO-переход"):
        decode_tags(((0, 3),), ("I-NAME",), "bio")
    with pytest.raises(ValueError, match="BIOES"):
        decode_tags(((0, 3),), ("B-NAME",), "bioes")
    with pytest.raises(ValueError, match="совпадать"):
        decode_tags(((0, 3),), (), "bio")
