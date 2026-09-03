"""Ограничения переходов и Viterbi для BIO/BIOES."""

from __future__ import annotations

from collections.abc import Sequence
from math import inf

from uzner.data.tagging import TagScheme, validate_scheme


def split_tag(tag: str) -> tuple[str, str | None]:
    """Разделяет корректно сформированный тег на префикс и класс."""
    if tag == "O":
        return "O", None
    prefix, separator, label = tag.partition("-")
    if separator != "-" or not prefix or not label:
        raise ValueError(f"Некорректный тег: {tag!r}")
    return prefix, label


def allowed_start(tag: str, scheme: TagScheme) -> bool:
    """Проверяет допустимость тега в начале последовательности."""
    validate_scheme(scheme)
    prefix, _ = split_tag(tag)
    if scheme == "bio":
        return prefix in {"O", "B"}
    return prefix in {"O", "B", "S"}


def allowed_end(tag: str, scheme: TagScheme) -> bool:
    """Проверяет допустимость тега в конце последовательности."""
    validate_scheme(scheme)
    prefix, _ = split_tag(tag)
    if scheme == "bio":
        return prefix in {"O", "B", "I"}
    return prefix in {"O", "E", "S"}


def allowed_transition(previous: str, current: str, scheme: TagScheme) -> bool:
    """Проверяет один переход в выбранной схеме разметки."""
    validate_scheme(scheme)
    previous_prefix, previous_label = split_tag(previous)
    current_prefix, current_label = split_tag(current)
    if scheme == "bio":
        if current_prefix == "I":
            return previous_prefix in {"B", "I"} and previous_label == current_label
        return current_prefix in {"O", "B"}
    if previous_prefix in {"B", "I"}:
        return current_prefix in {"I", "E"} and previous_label == current_label
    return current_prefix in {"O", "B", "S"}


def constrained_viterbi(
    emissions: Sequence[Sequence[float]],
    tags: Sequence[str],
    scheme: TagScheme,
    transition_scores: Sequence[Sequence[float]] | None = None,
) -> tuple[int, ...]:
    """Находит лучшую валидную последовательность индексов тегов."""
    validate_scheme(scheme)
    if not emissions:
        return ()
    tag_count = len(tags)
    if tag_count == 0 or any(len(row) != tag_count for row in emissions):
        raise ValueError("Размерность emissions не совпадает со словарём тегов")
    if transition_scores is not None and (
        len(transition_scores) != tag_count
        or any(len(row) != tag_count for row in transition_scores)
    ):
        raise ValueError("Матрица переходов должна иметь размер tags x tags")

    scores = [
        value if allowed_start(tags[index], scheme) else -inf
        for index, value in enumerate(emissions[0])
    ]
    backpointers: list[list[int]] = []
    for row in emissions[1:]:
        next_scores = [-inf] * tag_count
        pointers = [-1] * tag_count
        for current_index, emission in enumerate(row):
            for previous_index, previous_score in enumerate(scores):
                if not allowed_transition(tags[previous_index], tags[current_index], scheme):
                    continue
                transition = (
                    transition_scores[previous_index][current_index]
                    if transition_scores is not None
                    else 0.0
                )
                candidate = previous_score + transition + emission
                if candidate > next_scores[current_index]:
                    next_scores[current_index] = candidate
                    pointers[current_index] = previous_index
        scores = next_scores
        backpointers.append(pointers)

    allowed_final = [
        score if allowed_end(tags[index], scheme) else -inf for index, score in enumerate(scores)
    ]
    final_index = max(range(tag_count), key=allowed_final.__getitem__)
    if allowed_final[final_index] == -inf:
        raise ValueError("Для emissions не найдено валидной последовательности")
    path = [final_index]
    for pointers in reversed(backpointers):
        path.append(pointers[path[-1]])
    path.reverse()
    return tuple(path)
