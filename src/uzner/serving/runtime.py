"""Резидентный ансамбль: все три модели загружаются один раз на GPU."""

import json
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import torch

from uzner.config import ExperimentConfig
from uzner.data.windows import WindowCollator, build_window_features
from uzner.domain import Document, Prediction
from uzner.evaluation.span_vote import majority_vote
from uzner.models.token_tagger import TokenTagger
from uzner.posthoc.config import Variant
from uzner.posthoc.rules import dictionary_rule, repeat
from uzner.serving.bundle import SOURCES, load_lexicon, verify_bundle
from uzner.serving.config import RuntimeConfig
from uzner.training.checkpoint import load_model_checkpoint
from uzner.training.prediction import WindowLogits, aggregate_window_logits, decode_documents
from uzner.training.span_prediction import SpanWindowScores, decode_span_windows


@dataclass
class Component:
    """Веса, tokenizer и backend одного голоса ансамбля."""

    name: str
    model: TokenTagger
    tokenizer: Any
    experiment: ExperimentConfig
    engine: Any = None


class ResidentEnsemble:
    """Сохраняет исходные окна, majority 2/3 и порядок словаря и повторов."""

    def __init__(self, config: RuntimeConfig, *, verify: bool = True) -> None:
        """Загружает локальные ресурсы и запрещает скрытый CPU fallback."""
        if not torch.cuda.is_available():
            raise RuntimeError("Ансамбль требует NVIDIA CUDA GPU")
        self.config = config
        self.device = torch.device("cuda:0")
        torch.set_num_threads(config.cpu_threads)
        if verify:
            verify_bundle(config.bundle)
        self.lexicon = load_lexicon(config.bundle / "lexicon.json")
        self.components = tuple(self._load(name) for name, _ in SOURCES)
        # Освобождаем временные FP32 GPU-копии encoder-ов после загрузки engines.
        torch.cuda.empty_cache()
        self.variant = Variant("c02", "lex_add", normalized=True, propensity=0.9, repeat_after=True)
        self.timings: dict[str, float] = {}

    def _load(self, name: str) -> Component:
        """Восстанавливает исходный head; CRF на CPU исключает мелкие CUDA launches."""
        path = self.config.bundle / "models" / name
        experiment = ExperimentConfig.from_mapping(
            json.loads((path / "experiment_config.json").read_text("utf-8"))
        )
        loaded = load_model_checkpoint(path, experiment, self.device)
        loaded.model.eval().requires_grad_(False)
        if self.config.crf_cpu and getattr(loaded.model, "crf", None) is not None:
            loaded.model.crf.cpu()
            if self.config.crf_compile:
                loaded.model.crf.compile_cpu_decode()
        engine = None
        if name in self.config.engine_models:
            from uzner.serving.tensorrt_runtime import TensorRTEncoder

            engine_path = self.config.engine_dir / f"{name}.plan"
            # Нет скрытого fallback: состав гибридного исполнения задан конфигом.
            engine = TensorRTEncoder(engine_path, self.config.precision)
            loaded.model.encoder.cpu()
        return Component(name, loaded.model, loaded.tokenizer, experiment, engine)

    def _forward(self, component: Component, batch) -> torch.Tensor:
        """Исполняет encoder и прежнюю голову с явно выбранной точностью."""
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16
        with torch.autocast("cuda", dtype=dtype, enabled=self.config.precision != "fp32"):
            if component.engine is None:
                output = component.model(
                    batch.input_ids, batch.attention_mask, batch.token_type_ids
                )
                return output.logits
            hidden = component.engine(batch.input_ids, batch.attention_mask)
            return component.model.classifier(hidden)

    @torch.inference_mode()
    def _predict_component(
        self, component: Component, documents: tuple[Document, ...]
    ) -> tuple[Prediction, ...]:
        """Группирует окна без изменения offsets, затем применяет общий decoder."""
        started = perf_counter()
        features = build_window_features(
            documents,
            component.tokenizer,
            component.experiment.tokenization,
            component.experiment.model.tag_scheme,
            with_labels=False,
        )
        if self.config.sort_windows:
            features = tuple(sorted(features, key=lambda x: len(x.input_ids)))
        self.timings[f"{component.name}_tokenize"] = perf_counter() - started
        collate = WindowCollator(component.tokenizer.pad_token_id)
        windows: list[WindowLogits] = []
        span_windows: list[list[SpanWindowScores]] = [[] for _ in documents]
        forward_seconds = 0.0
        for start in range(0, len(features), self.config.window_batch):
            batch = collate(features[start : start + self.config.window_batch]).to(self.device)
            before = perf_counter()
            logits = self._forward(component, batch).detach().float().cpu()
            forward_seconds += perf_counter() - before
            if component.experiment.model.architecture == "span":
                probabilities = logits.sigmoid()
                for row, index, offsets in zip(
                    probabilities, batch.document_indices, batch.offsets, strict=True
                ):
                    width = len(offsets)
                    span_windows[index].append(
                        SpanWindowScores(offsets, row[:, :width, :width].clone())
                    )
            else:
                for row, index, offsets in zip(
                    logits, batch.document_indices, batch.offsets, strict=True
                ):
                    windows.append(WindowLogits(index, offsets, row[: len(offsets)]))
        self.timings[f"{component.name}_forward_transfer"] = forward_seconds
        before = perf_counter()
        if component.experiment.model.architecture == "span":
            result = tuple(
                decode_span_windows(doc.hash, scores, component.experiment.model.span_threshold)
                for doc, scores in zip(documents, span_windows, strict=True)
            )
        else:
            emissions = aggregate_window_logits(
                tuple(windows), len(documents), average_probabilities=False
            )
            result = decode_documents(documents, emissions, component.model)
        self.timings[f"{component.name}_decode"] = perf_counter() - before
        return result

    @torch.inference_mode()
    def predict_documents(self, documents: tuple[Document, ...]) -> tuple[Prediction, ...]:
        """Обрабатывает произвольные тексты без кэша ответов или обращения к gold."""
        if any(doc.entities for doc in documents):
            raise ValueError("Сервис не принимает gold-сущности")
        # Пустые/пробельные строки законны в API, но не имеют токенов для CRF.
        active = tuple(doc for doc in documents if doc.text.strip())
        if not active:
            return tuple(Prediction(doc.hash, ()) for doc in documents)
        sources = tuple(self._predict_component(c, active) for c in self.components)
        before = perf_counter()
        voted = majority_vote(active, sources)
        result = {}
        for doc, prediction in zip(active, voted, strict=True):
            prediction = dictionary_rule(doc, prediction, self.variant, self.lexicon)
            prediction = repeat(doc, prediction, replace(self.variant, normalized=False))
            Document(doc.hash, doc.text, prediction.entities)
            result[doc.hash] = prediction
        self.timings["vote_dictionary_repeats"] = perf_counter() - before
        return tuple(result.get(doc.hash, Prediction(doc.hash, ())) for doc in documents)

    def predict(self, texts: list[str]) -> tuple[tuple, ...]:
        """Адаптирует интерфейс текстов, не используя внешний hash как признак."""
        documents = tuple(Document(str(i), text, ()) for i, text in enumerate(texts))
        return tuple(pred.entities for pred in self.predict_documents(documents))
