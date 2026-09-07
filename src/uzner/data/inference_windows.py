"""Альтернативная нарезка только для независимой inference-абляции."""

from uzner.config import TokenizationConfig
from uzner.data.windows import WindowFeature, canonicalize_offsets
from uzner.domain import Document


def word_boundary_features(
    documents: tuple[Document, ...], tokenizer, config: TokenizationConfig
) -> tuple[WindowFeature, ...]:
    """Нарезает полную токенизацию у пробелов, сохраняя offsets и special tokens."""
    result = []
    for document_index, document in enumerate(documents):
        encoded = tokenizer(document.text, return_offsets_mapping=True, truncation=False)
        raw = encoded["offset_mapping"]
        content = [i for i, (a, b) in enumerate(raw) if b > a]
        if not content:
            raise ValueError("Нет содержательных токенов")
        first, stop = content[0], content[-1] + 1
        ids = encoded["input_ids"]
        body = list(range(first, stop))
        prefix, suffix = list(range(first)), list(range(stop, len(ids)))
        capacity = config.max_length - len(prefix) - len(suffix)
        if capacity <= config.stride:
            raise ValueError("Недостаточный размер окна")
        left, window = 0, 0
        while left < len(body):
            right = min(left + capacity, len(body))
            if right < len(body):
                for cut in range(right, max(left + config.stride + 1, right - 16), -1):
                    start = raw[body[cut]][0]
                    if start and document.text[start - 1].isspace():
                        right = cut
                        break
            selected = prefix + body[left:right] + suffix
            token_types = encoded.get("token_type_ids")
            result.append(
                WindowFeature(
                    document_index=document_index,
                    window_index=window,
                    input_ids=tuple(ids[i] for i in selected),
                    attention_mask=tuple(1 for _ in selected),
                    offsets=canonicalize_offsets(document.text, [raw[i] for i in selected]),
                    token_type_ids=None
                    if token_types is None
                    else tuple(token_types[i] for i in selected),
                    labels=None,
                )
            )
            if right == len(body):
                break
            next_left = max(left + 1, right - config.stride)
            for cut in range(next_left, max(left, next_left - 16), -1):
                start = raw[body[cut]][0]
                if start and document.text[start - 1].isspace():
                    next_left = cut
                    break
            left, window = next_left, window + 1
    return tuple(result)
