"""Потоковые проверки пересечений с official и дисковая exact-дедупликация."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from uzner.data.external_audit import normalized_text
from uzner.data.io import read_jsonl


def content_key(text: str) -> str:
    """Строит нормализованный ключ только для сравнения, не для замены текста."""
    return normalized_text(text)


def lexical_key(text: str) -> str:
    """Убирает пунктуационные различия только в представлении для поиска утечек."""
    return " ".join(re.findall(r"[^\W_]+", content_key(text), flags=re.UNICODE))


def digest(text: str) -> str:
    """Вычисляет стабильный SHA-256 строки."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Reference:
    """Один неизменяемый документ official для сравнения."""

    split: str
    identifier: str
    lexical: str


@dataclass(frozen=True, slots=True)
class Match:
    """Объяснимое совпадение или близкий дубликат контрольного текста."""

    reason: str
    reference: str


class LeakageIndex:
    """Находит полные совпадения, включённые фрагменты и близкие копии dev."""

    def __init__(self, official_root: Path) -> None:
        """Индексирует official train/dev, не меняя и не передавая их наружу."""
        import ahocorasick

        self.exact: dict[str, Reference] = {}
        self.references: list[Reference] = []
        patterns: dict[str, Reference] = {}
        anchors: defaultdict[str, list[int]] = defaultdict(list)
        for split in ("train", "dev"):
            for row in read_jsonl(official_root / f"{split}.jsonl"):
                key = lexical_key(row["text"])
                reference = Reference(split, row["hash"], key)
                self.exact[digest(content_key(row["text"]))] = reference
                self.exact[digest(key)] = reference
                self.references.append(reference)
                if len(key) >= 24 and len(key.split()) >= 4:
                    patterns[f" {key} "] = reference
                for sentence in re.split(r"[.!?\n]+", row["text"]):
                    part = lexical_key(sentence)
                    if len(part) >= 80 and len(part.split()) >= 10:
                        patterns[f" {part} "] = reference
                words = key.split()
                if split == "dev" and len(words) >= 12:
                    for start in (0, max(0, len(words) // 2 - 2), len(words) - 5):
                        anchors[" " + " ".join(words[start : start + 5]) + " "].append(
                            len(self.references) - 1
                        )
        self.contains = ahocorasick.Automaton()
        for key, reference in patterns.items():
            self.contains.add_word(key, reference)
        self.contains.make_automaton()
        self.near = ahocorasick.Automaton()
        for key, indices in anchors.items():
            if len(set(indices)) <= 5:
                self.near.add_word(key, tuple(set(indices)))
        self.near.make_automaton()

    def check(self, text: str) -> Match | None:
        """Возвращает тип пересечения; fuzzy-поиск ограничен official dev."""
        from rapidfuzz.fuzz import partial_ratio

        normalized = content_key(text)
        lexical = lexical_key(normalized)
        reference = self.exact.get(digest(normalized)) or self.exact.get(digest(lexical))
        if reference is not None:
            return Match(f"official_{reference.split}_exact", reference.identifier)
        padded = f" {lexical} "
        matched: Reference | None = None
        if len(self.contains):
            for _, reference in self.contains.iter(padded):
                matched = reference
                if reference.split == "dev":
                    break
        if matched is not None:
            return Match(f"official_{matched.split}_contained", matched.identifier)
        if len(lexical) < 80 or len(lexical.split()) < 12 or not len(self.near):
            return None
        candidates: set[int] = set()
        for _, indices in self.near.iter(padded):
            candidates.update(indices)
        for index in sorted(candidates):
            reference = self.references[index]
            if partial_ratio(lexical, reference.lexical, score_cutoff=96) >= 96:
                return Match("official_dev_near_96", reference.identifier)
        return None


class SeenTexts:
    """Хранит дедупликационный индекс на диске вместо всех текстов в памяти."""

    def __init__(self, path: Path) -> None:
        """Создаёт новый индекс; существующий файл намеренно не переиспользует."""
        if path.exists():
            raise FileExistsError(path)
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE seen (digest TEXT PRIMARY KEY, identifier TEXT)")

    def previous_or_add(self, key: str, identifier: str) -> str | None:
        """Возвращает первый дубликат или сохраняет новый ключ."""
        cursor = self.connection.execute("SELECT identifier FROM seen WHERE digest=?", (key,))
        previous = cursor.fetchone()
        if previous:
            return str(previous[0])
        self.connection.execute("INSERT INTO seen VALUES (?, ?)", (key, identifier))
        return None

    def close(self) -> None:
        """Фиксирует завершённый индекс и закрывает соединение."""
        self.connection.commit()
        self.connection.close()
