"""Проверки train-only статистики и Unicode-safe поиска словарных форм."""

from uzner.domain import Document, Entity
from uzner.posthoc.lexicon import build_lexicon, surface_key, tokenize


def test_background_occurrences_reduce_propensity():
    """Неразмеченные вхождения не должны считаться надёжными положительными примерами."""
    train = (
        Document("t1", "Ali Ali", (Entity(0, 3, "NAME"),)),
        Document("t2", "Ali", (Entity(0, 3, "NAME"),)),
    )
    lexicon = build_lexicon(train)
    entry = lexicon.get("Ali")
    assert entry.support == 2
    assert entry.annotated == 2 and entry.occurrences == 3
    assert entry.propensity == 2 / 3 and entry.purity == 1
    assert lexicon.get("newname") is None


def test_casefold_expansion_preserves_original_offsets():
    """Расширение ß при casefold не сдвигает координаты исходного слова."""
    train = (Document("t", "STRASSE", (Entity(0, 7, "ORG"),)),)
    lexicon = build_lexicon(train, True)
    matches = tuple(lexicon.matches("😀 Straße!"))
    assert [(m.start, m.end) for m in matches] == [(2, 8)]
    assert "😀 Straße!"[matches[0].start : matches[0].end] == "Straße"


def test_apostrophes_whitespace_and_punctuation():
    """Варианты апострофа и пробелов совпадают лишь через токенные ключи."""
    assert surface_key("O‘zbekiston", True) == surface_key("O'ZBEKISTON", True)
    train = (Document("t", "New York", (Entity(0, 8, "GEO"),)),)
    lexicon = build_lexicon(train)
    matches = tuple(lexicon.matches("xNew York New\nYork."))
    assert [(m.start, m.end) for m in matches] == [(10, 18)]
    assert tokenize("😀 A")[1].start == 2


def test_support_counts_documents_not_repetitions():
    """Повторы одной формы в одном документе не увеличивают документную поддержку."""
    doc = Document("t", "Ali Ali", (Entity(0, 3, "NAME"), Entity(4, 7, "GEO")))
    entry = build_lexicon((doc,)).get("Ali")
    assert entry.support == 1 and entry.purity == 0.5
    assert entry.propensity == 0.5


def test_partial_token_gold_not_in_dictionary():
    """Разметка внутри цельного слова не обучает некорректную словарную форму."""
    assert not build_lexicon((Document("t", "Alining", (Entity(0, 3, "NAME"),)),)).entries
