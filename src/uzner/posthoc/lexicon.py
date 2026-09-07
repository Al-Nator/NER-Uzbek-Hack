"""Train-only словарь с учётом неразмеченных вхождений и исходных координат."""

import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from uzner.domain import Document

APOSTROPHES = "'’ʻʼ‘`ʼ＇"
WORDS = re.compile(r"[^\W_]+(?:['’ʻʼ‘`＇-][^\W_]+)*|[^\s]", re.UNICODE)
TRANSLATION = str.maketrans({char: "'" for char in APOSTROPHES})
Key = tuple[str, ...]


@dataclass(frozen=True)
class Token:
    """Ключ токена с неизменяемыми позициями в исходной строке."""

    value: str
    start: int
    end: int


def tokenize(text: str, normalized: bool = False) -> tuple[Token, ...]:
    """Нормализует только ключ поиска, никогда не координаты или исходный текст."""
    return tuple(
        Token(
            m.group().casefold().translate(TRANSLATION) if normalized else m.group(),
            m.start(),
            m.end(),
        )
        for m in WORDS.finditer(text)
    )


def surface_key(text: str, normalized: bool = False) -> Key:
    """Создаёт ключ формы; пробельные разделители не меняют состав токенов."""
    return tuple(token.value for token in tokenize(text, normalized))


@dataclass(frozen=True)
class LexiconEntry:
    """Метка, число документов и доля размеченных вхождений формы в train."""

    label: str
    support: int
    annotated: int
    occurrences: int
    purity: float

    @property
    def propensity(self) -> float:
        """Возвращает долю всех вхождений, размеченных выбранной меткой."""
        return self.annotated / self.occurrences if self.occurrences else 0.0


@dataclass(frozen=True)
class LexiconMatch:
    """Вхождение словарной формы с точными границами исходного текста."""

    start: int
    end: int
    entry: LexiconEntry


@dataclass(frozen=True)
class Lexicon:
    """Индекс форм исключительно из train, без чтения dev gold при применении."""

    entries: dict[Key, LexiconEntry]
    prefixes: frozenset[Key]
    normalized: bool
    max_tokens: int = 12

    def matches(self, text: str) -> Iterator[LexiconMatch]:
        """Ищет токенные формы, сохраняя оригинальные Unicode offsets."""
        for key, start, end in scan_keys(text, self.prefixes, self.normalized, self.max_tokens):
            entry = self.entries.get(key)
            if entry is not None:
                yield LexiconMatch(start, end, entry)

    def get(self, text: str) -> LexiconEntry | None:
        """Находит статистику целой формы, а не произвольной подстроки."""
        return self.entries.get(surface_key(text, self.normalized))


def scan_keys(
    text: str, prefixes: frozenset[Key], normalized: bool, max_tokens: int
) -> Iterator[tuple[Key, int, int]]:
    """Перебирает только известные префиксы, не сканируя весь словарь на каждом слове."""
    tokens = tokenize(text, normalized)
    for index, first in enumerate(tokens):
        key: Key = ()
        for last in tokens[index : index + max_tokens]:
            key += (last.value,)
            if key not in prefixes:
                break
            yield key, first.start, last.end


def build_lexicon(train: tuple[Document, ...], normalized: bool = False) -> Lexicon:
    """Считает метки и фоновые вхождения только по переданному обучающему набору."""
    counts: dict[Key, Counter] = defaultdict(Counter)
    documents: dict[tuple[Key, str], set[str]] = defaultdict(set)
    for document in train:
        tokens = tokenize(document.text, normalized)
        starts, ends = {t.start for t in tokens}, {t.end for t in tokens}
        for entity in document.entities:
            key = surface_key(document.text[entity.start : entity.end], normalized)
            if not key or len(key) > 12 or entity.start not in starts or entity.end not in ends:
                continue
            counts[key][entity.label] += 1
            documents[key, entity.label].add(document.hash)
    prefixes = frozenset(key[:n] for key in counts for n in range(1, len(key) + 1))
    occurrences: Counter = Counter()
    for document in train:
        for key, _, _ in scan_keys(document.text, prefixes, normalized, 12):
            if key in counts:
                occurrences[key] += 1
    entries = {}
    for key, labels in counts.items():
        label = min(labels, key=lambda item: (-labels[item], item))
        entries[key] = LexiconEntry(
            label,
            len(documents[key, label]),
            labels[label],
            occurrences[key],
            labels[label] / sum(labels.values()),
        )
    return Lexicon(entries, prefixes, normalized)
