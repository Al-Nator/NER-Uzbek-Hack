"""Общий streaming-инференс фиксированного GP-checkpoint для research-веток."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from uzner.config import ExperimentConfig
from uzner.data.windows import WindowFeature, build_window_features
from uzner.domain import Document, Prediction
from uzner.training.checkpoint import load_model_checkpoint
from uzner.training.data_setup import make_loader
from uzner.training.span_prediction import SpanPredictionAccumulator, SpanWindowScores


@dataclass(frozen=True)
class FrozenWindow:
    """Контекст и scores одного окна без использования gold при инференсе."""

    document_index: int
    feature: WindowFeature
    hidden: torch.Tensor
    probabilities: torch.Tensor


class FrozenPredictor:
    """Повторно использует encoder/GP и общий decoder из выбранного best."""

    def __init__(self, source: Path) -> None:
        """Загружает единственный frozen checkpoint, не копируя его веса."""
        self.config = ExperimentConfig.from_mapping(
            json.loads((source / "checkpoints/best/experiment_config.json").read_text("utf-8"))
        )
        loaded = load_model_checkpoint(
            source / "checkpoints/best", self.config, torch.device("cuda")
        )
        self.model, self.tokenizer = loaded.model.eval(), loaded.tokenizer
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def windows(
        self, documents: tuple[Document, ...], *, with_labels: bool = False
    ) -> Iterator[FrozenWindow]:
        """Читает документы блоками; золото допустимо только для train-datastore."""
        for offset in range(0, len(documents), 128):
            group = documents[offset : offset + 128]
            features = build_window_features(
                group, self.tokenizer, self.config.tokenization, "bioes", with_labels=with_labels
            )
            loader = make_loader(
                features, self.tokenizer, batch_size=8, shuffle=False, seed=42, num_workers=0
            )
            cursor = 0
            for batch in loader:
                batch = batch.to(torch.device("cuda"))
                inputs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
                if batch.token_type_ids is not None:
                    inputs["token_type_ids"] = batch.token_type_ids
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = self.model.encoder(**inputs).last_hidden_state
                    probabilities = self.model.classifier(hidden).float().sigmoid()
                for row in range(len(batch.document_indices)):
                    feature = features[cursor]
                    n = len(feature.offsets)
                    yield FrozenWindow(
                        offset + feature.document_index,
                        feature,
                        hidden[row, :n].detach(),
                        probabilities[row, :, :n, :n].detach(),
                    )
                    cursor += 1

    def predict(self, documents: tuple[Document, ...]) -> tuple[Prediction, ...]:
        """Возвращает обычные exact-span предсказания без изменения scoring."""
        accumulator = SpanPredictionAccumulator(
            self.config.model.span_threshold, tuple(d.hash for d in documents), [], []
        )
        for window in self.windows(documents):
            accumulator.add(
                window.document_index,
                SpanWindowScores(window.feature.offsets, window.probabilities.cpu()),
            )
        return accumulator.finish()
