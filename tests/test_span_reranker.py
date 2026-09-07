"""Защита от утечки gold, корректность голов и выбор только на meta-valid."""

from dataclasses import replace

import pytest
import torch

from uzner.data.reranker_candidates import candidate_targets, candidates_from_sources
from uzner.domain import Document, Entity, Prediction
from uzner.evaluation.metrics import evaluate_predictions
from uzner.models.span_reranker import RerankerConfig, SpanReranker
from uzner.training.reranker_fit import fit_reranker, make_reranker_dataset


def sample_dataset():
    """Создаёт небольшой набор с истинными, ложными и пересекающимися кандидатами."""
    documents = (Document("a", "Ali Toshkent", (Entity(0, 3, "NAME"),)), Document("b", "Ўз", ()))
    first = (Prediction("a", (Entity(0, 3, "NAME"), Entity(4, 12, "GEO"))), Prediction("b", ()))
    second = (Prediction("a", (Entity(0, 12, "ORG"),)), Prediction("b", (Entity(0, 2, "GEO"),)))
    return documents, (first, first, second)


def test_candidates_are_gold_free_and_unicode_safe():
    """Gold влияет только на цели, а не на состав кандидатов или признаки."""
    documents, sources = sample_dataset()
    candidates = candidates_from_sources(documents, sources)
    empty = tuple(replace(d, entities=()) for d in documents)
    assert candidates == candidates_from_sources(empty, sources)
    assert sum(candidate_targets(candidates, documents)) == 1
    assert sum(candidate_targets(candidates, empty)) == 0
    assert all(len(c.characters) == 130 for c in candidates)
    assert all(0 <= x < 4096 for c in candidates for x in c.characters)
    with pytest.raises(ValueError):
        candidates_from_sources(documents, sources[:2])


@pytest.mark.parametrize("variant", ["linear", "mlp", "char_context"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_heads_and_fit_select_only_meta_valid(tmp_path, variant, device):
    """Все головы обучаются; изменение official gold не меняет выбранные веса."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA недоступна для короткого GPU-smoke")
    documents, sources = sample_dataset()
    dataset = make_reranker_dataset(documents, sources)
    model = SpanReranker(dataset.features.shape[1], variant)
    assert torch.isfinite(model(dataset.features, dataset.characters)).all()

    class TestRun:
        """Изолирует тестовый журнал от настоящего MLflow."""

        def __init__(self, root):
            """Создаёт временную папку только для тестовых весов."""
            self.root = root
            root.mkdir()

        def log(self, values, step=0):
            """Не публикует тестовые метрики наружу."""

        def evaluate(self, gold, predictions, train, prefix):
            """Проверяет exact-контракт без сохранения настоящего эксперимента."""
            return evaluate_predictions(gold, predictions).micro.f1

    config = RerankerConfig("test", variant, epochs=2, batch_size=2)
    changed = make_reranker_dataset(tuple(replace(d, entities=()) for d in documents), sources)
    for index, external in enumerate((dataset, changed)):
        fit_reranker(
            config,
            dataset,
            dataset,
            {"dev": external},
            documents,
            TestRun(tmp_path / str(index)),
            torch.device(device),
        )
    left = torch.load(tmp_path / "0/checkpoints/best.pt", weights_only=False)
    right = torch.load(tmp_path / "1/checkpoints/best.pt", weights_only=False)
    assert left["threshold"] == right["threshold"]
    assert all(torch.equal(left["model"][key], right["model"][key]) for key in left["model"])


def test_reranker_config_rejects_invalid_values():
    """Неизвестные варианты и невалидные пороги запрещены."""
    with pytest.raises(ValueError):
        RerankerConfig("test", "unknown")
    with pytest.raises(ValueError):
        RerankerConfig("test", "linear", thresholds=(1.0,))
