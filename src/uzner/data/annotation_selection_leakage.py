"""Fallback official leakage index для окружений без optional pyahocorasick."""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from uzner.data.curation.leakage import Match, content_key, digest, lexical_key


def iter_jsonl(path: Path):
    """Потоково читает JSONL для fallback."""
    import json

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class SimpleLeakage:
    """Pure-Python exact/containment/near index с n-gram anchors."""

    def __init__(self, root: Path) -> None:
        """Строит индекс official train/dev."""
        self.exact: dict[str, Match] = {}
        self.ngrams: dict[str, Match] = {}
        self.near: defaultdict[str, list[tuple[Match, str]]] = defaultdict(list)
        for split in ("train", "dev"):
            for row in iter_jsonl(root / f"{split}.jsonl"):
                text, value = str(row["text"]), lexical_key(str(row["text"]))
                match = Match(f"official_{split}_exact", str(row["hash"]))
                self.exact[digest(content_key(text))] = match
                self.exact[digest(value)] = match
                words = value.split()
                for index in range(max(0, len(words) - 9)):
                    self.ngrams.setdefault(
                        " ".join(words[index : index + 10]),
                        Match(f"official_{split}_contained", str(row["hash"])),
                    )
                if split == "dev" and len(words) >= 12:
                    for index in (0, len(words) // 2, len(words) - 5):
                        self.near[" ".join(words[index : index + 5])].append((match, value))

    def check(self, text: str) -> Match | None:
        """Проверяет exact, containment и near anchors."""
        normalized, value = content_key(text), lexical_key(text)
        hit = self.exact.get(digest(normalized)) or self.exact.get(digest(value))
        if hit is not None:
            return hit
        words = value.split()
        for index in range(max(0, len(words) - 9)):
            hit = self.ngrams.get(" ".join(words[index : index + 10]))
            if hit is not None:
                return hit
        refs: dict[str, tuple[Match, str]] = {}
        for index in (0, len(words) // 2, len(words) - 5) if len(words) >= 12 else ():
            for item in self.near.get(" ".join(words[index : index + 5]), ()):
                refs[item[0].reference] = item
        for match, reference in refs.values():
            if SequenceMatcher(None, value, reference).ratio() >= 0.96:
                return Match("official_dev_near_96", match.reference)
        return None
