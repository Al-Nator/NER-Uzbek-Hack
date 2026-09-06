"""Типизированные YAML-конфигурации данных и экспериментов."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from uzner.config_helpers import check_keys as _check_keys
from uzner.config_helpers import load_yaml as _load_yaml
from uzner.config_helpers import require_mapping as _require_mapping
from uzner.data.sources import (
    DataConfig,
    DataSourceConfig,
    SourceKind,
    Split,
    resolve_sources,
)
from uzner.data.tagging import TagScheme, validate_scheme
from uzner.research_config import SpanResearchConfig

__all__ = [
    "DataConfig",
    "DataSourceConfig",
    "EncoderConfig",
    "ExperimentConfig",
    "ModelConfig",
    "SourceKind",
    "Split",
    "TokenizationConfig",
    "TrainingConfig",
    "load_data_config",
    "load_experiment_config",
    "resolve_sources",
    "with_run_suffix",
]


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Точная ссылка на pretrained encoder."""

    name: str
    revision: str
    trust_remote_code: bool = False
    use_safetensors: bool | None = None

    def __post_init__(self) -> None:
        """Запрещает неполное описание encoder-а."""
        if not self.name or not self.revision:
            raise ValueError("Encoder требует непустые name и revision")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EncoderConfig:
        """Создаёт конфигурацию encoder-а из YAML-объекта."""
        _check_keys(
            value,
            {"name", "revision", "trust_remote_code", "use_safetensors"},
            name="encoder",
        )
        use_safetensors = value.get("use_safetensors")
        if use_safetensors is not None and not isinstance(use_safetensors, bool):
            raise ValueError("encoder.use_safetensors должен быть bool или null")
        return cls(
            name=value.get("name"),
            revision=str(value.get("revision", "main")),
            trust_remote_code=bool(value.get("trust_remote_code", False)),
            use_safetensors=use_safetensors,
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Независимые настройки постановки token-level NER."""

    architecture: str
    tag_scheme: TagScheme
    head: Literal["softmax", "crf", "biaffine", "global_pointer"]
    decoder: Literal["greedy", "constrained", "crf", "span"]
    dropout: float = 0.1
    span_head_size: int = 64
    span_threshold: float = 0.5
    research: SpanResearchConfig = field(default_factory=SpanResearchConfig)

    def __post_init__(self) -> None:
        """Проверяет совместимость головы и декодера."""
        if self.research.enabled and (self.architecture != "span" or self.head != "global_pointer"):
            raise ValueError("Research losses поддерживаются только для GlobalPointer")
        if self.architecture == "span":
            if self.head not in {"biaffine", "global_pointer"} or self.decoder != "span":
                raise ValueError("Span architecture требует biaffine/global_pointer и span decoder")
            if self.span_head_size < 2 or self.span_head_size % 2:
                raise ValueError("span_head_size должен быть положительным чётным числом")
            if not 0 < self.span_threshold < 1 or not 0 <= self.dropout < 1:
                raise ValueError("Некорректные span_threshold или dropout")
            return
        if self.architecture != "token_tagging":
            raise ValueError("Поддерживаются architecture=token_tagging и span")
        validate_scheme(self.tag_scheme)
        if self.head not in {"softmax", "crf"}:
            raise ValueError(f"Некорректная head: {self.head!r}")
        if self.decoder not in {"greedy", "constrained", "crf"}:
            raise ValueError(f"Некорректный decoder: {self.decoder!r}")
        if (self.head == "crf") != (self.decoder == "crf"):
            raise ValueError("CRF-head должна использовать CRF-decoder и наоборот")
        if self.tag_scheme == "bioes" and self.decoder == "greedy":
            raise ValueError("BIOES требует constrained или CRF decoder")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout должен лежать в диапазоне [0, 1)")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ModelConfig:
        """Создаёт конфигурацию модели из YAML-объекта."""
        allowed = {
            "architecture",
            "tag_scheme",
            "head",
            "decoder",
            "dropout",
            "span_head_size",
            "span_threshold",
            "research",
        }
        _check_keys(value, allowed, name="model")
        return cls(
            architecture=str(value.get("architecture", "token_tagging")),
            tag_scheme=value.get("tag_scheme", "bio"),
            head=value.get("head", "softmax"),
            decoder=value.get("decoder", "greedy"),
            dropout=float(value.get("dropout", 0.1)),
            span_head_size=int(value.get("span_head_size", 64)),
            span_threshold=float(value.get("span_threshold", 0.5)),
            research=SpanResearchConfig.from_mapping(value.get("research", {})),
        )


@dataclass(frozen=True, slots=True)
class TokenizationConfig:
    """Параметры окон и перекрытия для одного encoder-а."""

    max_length: int = 512
    stride: int = 128

    def __post_init__(self) -> None:
        """Проверяет длину контекста и перекрытие окон."""
        if self.max_length < 8:
            raise ValueError("max_length должен быть не меньше 8")
        if not 0 <= self.stride < self.max_length:
            raise ValueError("stride должен удовлетворять 0 <= stride < max_length")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TokenizationConfig:
        """Создаёт параметры токенизации из YAML-объекта."""
        _check_keys(value, {"max_length", "stride"}, name="tokenization")
        return cls(
            max_length=int(value.get("max_length", 512)),
            stride=int(value.get("stride", 128)),
        )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Параметры воспроизводимого обучения."""

    seed: int = 42
    epochs: int = 5
    batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    head_learning_rate: float | None = None
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    bf16: bool = True
    num_workers: int = 0
    require_gpu: bool = True
    gradient_checkpointing: bool = False
    optimizer: Literal["adamw", "adamw_8bit"] = "adamw"
    log_every_steps: int = 25
    early_stopping_patience: int = 2
    initial_checkpoint: str | None = None

    def __post_init__(self) -> None:
        """Проверяет численные параметры обучения."""
        positive = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "max_grad_norm": self.max_grad_norm,
            "log_every_steps": self.log_every_steps,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Параметры должны быть положительными: {invalid}")
        if self.head_learning_rate is not None and not 0 < self.head_learning_rate < float("inf"):
            raise ValueError("head_learning_rate должен быть конечным и положительным")
        if self.weight_decay < 0 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("Некорректные weight_decay или warmup_ratio")
        if self.num_workers < 0:
            raise ValueError("num_workers не может быть отрицательным")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience не может быть отрицательным")
        if self.optimizer not in {"adamw", "adamw_8bit"}:
            raise ValueError("optimizer должен быть adamw или adamw_8bit")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrainingConfig:
        """Создаёт параметры обучения из YAML-объекта."""
        allowed = {
            "seed",
            "epochs",
            "batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "head_learning_rate",
            "weight_decay",
            "warmup_ratio",
            "max_grad_norm",
            "bf16",
            "num_workers",
            "require_gpu",
            "gradient_checkpointing",
            "optimizer",
            "log_every_steps",
            "early_stopping_patience",
            "initial_checkpoint",
        }
        _check_keys(value, allowed, name="training")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Полная конфигурация одного сравнимого запуска."""

    run_id: str
    data_config: str
    encoder: EncoderConfig
    model: ModelConfig
    tokenization: TokenizationConfig
    training: TrainingConfig
    pipeline: Literal["official_reference", "uzner"] = "uzner"
    output_root: str = "runs"
    primary_metric: Literal["exact_micro_f1", "dev_loss"] = "exact_micro_f1"

    def __post_init__(self) -> None:
        """Проверяет идентификатор и критерий выбора модели."""
        if not self.run_id or any(character.isspace() for character in self.run_id):
            raise ValueError("run_id должен быть непустым и не содержать пробелы")
        if not self.data_config or not self.output_root:
            raise ValueError("Нужны data_config и output_root")
        if self.pipeline not in {"official_reference", "uzner"}:
            raise ValueError(f"Неизвестный pipeline: {self.pipeline!r}")
        expected_metric = "dev_loss" if self.pipeline == "official_reference" else "exact_micro_f1"
        if self.primary_metric != expected_metric:
            raise ValueError(f"Для pipeline={self.pipeline} нужен primary_metric={expected_metric}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExperimentConfig:
        """Создаёт полный эксперимент из YAML-объекта."""
        allowed = {
            "run_id",
            "data_config",
            "encoder",
            "model",
            "tokenization",
            "training",
            "pipeline",
            "output_root",
            "primary_metric",
        }
        _check_keys(value, allowed, name="experiment")
        return cls(
            run_id=value.get("run_id"),
            data_config=value.get("data_config"),
            encoder=EncoderConfig.from_mapping(
                _require_mapping(value.get("encoder"), name="encoder")
            ),
            model=ModelConfig.from_mapping(_require_mapping(value.get("model"), name="model")),
            tokenization=TokenizationConfig.from_mapping(
                _require_mapping(value.get("tokenization", {}), name="tokenization")
            ),
            training=TrainingConfig.from_mapping(
                _require_mapping(value.get("training", {}), name="training")
            ),
            pipeline=value.get("pipeline", "uzner"),
            output_root=str(value.get("output_root", "runs")),
            primary_metric=str(value.get("primary_metric", "exact_micro_f1")),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Возвращает конфигурацию как сериализуемое отображение."""
        return asdict(self)


def load_data_config(path: Path) -> DataConfig:
    """Загружает конфигурацию источников данных."""
    value = _load_yaml(path)
    _check_keys(value, {"data"}, name=str(path))
    return DataConfig.from_mapping(_require_mapping(value.get("data"), name="data"))


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Загружает конфигурацию одного эксперимента."""
    value = _load_yaml(path)
    _check_keys(value, {"experiment"}, name=str(path))
    return ExperimentConfig.from_mapping(
        _require_mapping(value.get("experiment"), name="experiment")
    )


def with_run_suffix(config: ExperimentConfig, suffix: str | None) -> ExperimentConfig:
    """Добавляет безопасный suffix для повторного запуска без перезаписи run-а."""
    if suffix is None:
        return config
    if not suffix or any(not (character.isalnum() or character in "-_.") for character in suffix):
        raise ValueError("run suffix допускает только буквы, цифры, '-', '_' и '.'")
    return replace(config, run_id=f"{config.run_id}_{suffix}")
