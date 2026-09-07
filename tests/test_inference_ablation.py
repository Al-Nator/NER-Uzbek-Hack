"""Проверки coverage альтернативных окон и ненулевых отрицательных голосов."""

import torch
from transformers import AutoTokenizer

from tests.test_training_pipeline import _tiny_project
from uzner.config import load_experiment_config
from uzner.data.inference_windows import word_boundary_features
from uzner.domain import Document
from uzner.training.span_prediction import SpanWindowScores, decode_span_windows


def test_word_windows_preserve_all_offsets_and_special_tokens(tmp_path):
    """Длинный документ не обрезается после двух окон, Unicode-координаты полные."""
    config = load_experiment_config(_tiny_project(tmp_path))
    tokenizer = AutoTokenizer.from_pretrained(tmp_path / "tiny_model")
    document = Document("a", "Ali Тошкент " * 40)
    windows = word_boundary_features((document,), tokenizer, config.tokenization)
    raw = tokenizer(document.text, return_offsets_mapping=True)["offset_mapping"]
    assert len(windows) > 2
    expected = {tuple(o) for o in raw if o[1] > o[0]}
    actual = {o for w in windows for o in w.offsets if o[1] > o[0]}
    assert actual == expected
    assert all(len(w.input_ids) <= config.tokenization.max_length for w in windows)
    assert all(
        w.input_ids[0] == tokenizer.cls_token_id and w.input_ids[-1] == tokenizer.sep_token_id
        for w in windows
    )


def test_center_weighting_keeps_negative_votes():
    """Отрицательное центральное окно может перевесить положительный край."""
    central = ((0, 0), *[(i, i + 1) for i in range(9)], (0, 0))
    edge = ((0, 0), *[(i, i + 1) for i in range(4, 13)], (0, 0))
    low = torch.zeros(3, len(central), len(central))
    high = torch.zeros(3, len(edge), len(edge))
    low[0, 5, 5], high[0, 1, 1] = 0.2, 0.9
    uniform = decode_span_windows(
        "a", [SpanWindowScores(central, low), SpanWindowScores(edge, high)], 0.5
    )
    weighted = decode_span_windows(
        "a", [SpanWindowScores(central, low, "center"), SpanWindowScores(edge, high, "center")], 0.5
    )
    assert len(uniform.entities) == 1
    assert not weighted.entities
