"""MELM-inspired примеры и замена одной сущности с точными character offsets."""

import hashlib
import random
import re
from dataclasses import dataclass

from uzner.domain import Document, Entity


@dataclass(frozen=True)
class MelmExample:
    """Одно train-упоминание, защищённое маркерами BIO на уровне слов."""

    source_hash: str
    entity_index: int
    input_ids: tuple[int, ...]
    entity_positions: tuple[int, ...]
    word_positions: tuple[tuple[int, ...], ...]


def make_examples(
    documents: tuple[Document, ...], tokenizer: object, *, max_length: int = 256, seed: int = 42
) -> tuple[MelmExample, ...]:
    """Выбирает до двух упоминаний на документ; контекст не используется как цель MLM."""
    rng = random.Random(seed)
    examples = []
    for doc in documents:
        indices = list(range(len(doc.entities)))
        rng.shuffle(indices)
        for index in indices[:2]:
            entity = doc.entities[index]
            words = re.findall(r"\S+", doc.text[entity.start : entity.end])
            ids, positions, word_positions = [], [], []
            for i, word in enumerate(words):
                marker = tokenizer.convert_tokens_to_ids(
                    f"<{'B' if i == 0 else 'I'}-{entity.label}>"
                )
                tokens = tokenizer.encode(word, add_special_tokens=False)
                ids.append(marker)
                wp = tuple(range(len(ids), len(ids) + len(tokens)))
                word_positions.append(wp)
                positions.extend(wp)
                ids.extend(tokens)
                ids.append(marker)
            if not positions or len(ids) > max_length - 16:
                continue
            budget = (max_length - len(ids) - 2) // 2
            left = tokenizer.encode(
                doc.text[max(0, entity.start - 512) : entity.start], add_special_tokens=False
            )[-budget:]
            right = tokenizer.encode(
                doc.text[entity.end : entity.end + 512], add_special_tokens=False
            )[:budget]
            shift = len(left) + 1
            examples.append(
                MelmExample(
                    doc.hash,
                    index,
                    tuple([tokenizer.bos_token_id] + left + ids + right + [tokenizer.eos_token_id]),
                    tuple(p + shift for p in positions),
                    tuple(tuple(p + shift for p in ps) for ps in word_positions),
                )
            )
    return tuple(examples)


def replace_entity(document: Document, index: int, surface: str, *, variant: str) -> Document:
    """Заменяет только выбранный span, сдвигая все последующие сущности."""
    old = document.entities[index]
    if not surface or surface != surface.strip() or any(c in surface for c in "\n\r<>"):
        raise ValueError("Некорректная поверхность MELM")
    text = document.text[: old.start] + surface + document.text[old.end :]
    delta = len(surface) - (old.end - old.start)
    entities = []
    for i, entity in enumerate(document.entities):
        if i == index:
            entities.append(Entity(old.start, old.start + len(surface), old.label))
        elif entity.start >= old.end:
            entities.append(Entity(entity.start + delta, entity.end + delta, entity.label))
        else:
            entities.append(entity)
    digest = hashlib.sha256(f"{document.hash}:{index}:{variant}:{text}".encode()).hexdigest()
    return Document("melm:" + digest, text, tuple(entities), "melm")


def preserve_suffix(original: str, generated: str) -> bool:
    """Требует сохранения явно распознанного узбекского окончания и письменности."""
    from uzner.evaluation.slices import detect_script

    if detect_script(original) != detect_script(generated):
        return False
    suffix = re.search(r"(ning|dan|dagi|ga|ni|da|нинг|дан|даги|га|ни|да)$", original.casefold())
    return suffix is None or generated.casefold().endswith(suffix.group())
