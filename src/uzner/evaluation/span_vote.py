"""Фиксированное голосование полных character spans без подбора по dev."""

from collections import Counter
from dataclasses import dataclass

from uzner.domain import Document, Entity, Prediction


@dataclass(frozen=True)
class SpanVoteConfig:
    """Три разных источника и строгое большинство, без калибровки scores."""

    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        """Исключает дублирование голоса одного run-а."""
        if len(self.sources) != 3 or len(set(self.sources)) != 3:
            raise ValueError("Нужны три разных источника")


def majority_vote(
    documents: tuple[Document, ...], sources: tuple[tuple[Prediction, ...], ...]
) -> tuple[Prediction, ...]:
    """Оставляет spans с двумя голосами; не читает gold-сущности документов."""
    if len(sources) != 3:
        raise ValueError("Нужны ровно три набора предсказаний")
    expected = {doc.hash for doc in documents}
    mappings = []
    for predictions in sources:
        mapping = {pred.hash: pred for pred in predictions}
        if len(mapping) != len(predictions) or set(mapping) != expected:
            raise ValueError("Неполное покрытие hash или повторный документ")
        mappings.append(mapping)
    result = []
    for doc in documents:
        votes = Counter()
        for mapping in mappings:
            entities = mapping[doc.hash].entities
            # Входные системы плоские: тогда большинство тоже не пересекается.
            Document(doc.hash, doc.text, entities)
            votes.update((e.start, e.end, e.label) for e in entities)
        entities = tuple(
            Entity(a, b, label, count / 3) for (a, b, label), count in votes.items() if count >= 2
        )
        Document(doc.hash, doc.text, entities)
        result.append(Prediction(doc.hash, entities))
    return tuple(result)
