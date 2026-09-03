"""Типизированные YAML-конфигурации данных и экспериментов."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
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

__all__ = [
    "DataConfig",
    "DataSourceConfig",
    "EncoderConfig",
    "ExperimentConfig",
    "ModelConfig",
    "SourceKind",
    "Split",
    "TrainingConfig",
    "load_data_config",
    "load_experiment_config",
    "resolve_sources",
]


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Точная ссылка на pretrained encoder."""

    name: str
    revision: str
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        """Запрещает неполное описание encoder-а."""
        if not self.name or not self.revision:
            raise ValueError("Encoder требует непустые name и revision")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EncoderConfig:
        """Создаёт конфигурацию encoder-а из YAML-объекта."""
        _check_keys(value, {"name", "revision", "trust_remote_code"}, name="encoder")
        return cls(
            name=value.get("name"),
            revision=str(value.get("revision", "main")),
            trust_remote_code=bool(value.get("trust_remote_code", False)),
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Независимые настройки постановки token-level NER."""

    architecture: str
    tag_scheme: TagScheme
    head: Literal["softmax", "crf"]
    decoder: Literal["greedy", "constrained", "crf"]
    dropout: float = 0.1

    def __post_init__(self) -> None:
        """Проверяет совместимость головы и декодера."""
        if self.architecture != "token_tagging":
            raise ValueError("Пока реализована только architecture=token_tagging")
        validate_scheme(self.tag_scheme)
        if self.head not in {"softmax", "crf"}:
            raise ValueError(f"Некорректная head: {self.head!r}")
        if self.decoder not in {"greedy", "constrained", "crf"}:
            raise ValueError(f"Некорректный decoder: {self.decoder!r}")
        if (self.head == "crf") != (self.decoder == "crf"):
            raise ValueError("CRF-head должна использовать CRF-decoder и наоборот")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout должен лежать в диапазоне [0, 1)")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ModelConfig:
        """Создаёт конфигурацию модели из YAML-объекта."""
        allowed = {"architecture", "tag_scheme", "head", "decoder", "dropout"}
        _check_keys(value, allowed, name="model")
        return cls(
            architecture=str(value.get("architecture", "token_tagging")),
            tag_scheme=value.get("tag_scheme", "bio"),
            head=value.get("head", "softmax"),
            decoder=value.get("decoder", "greedy"),
            dropout=float(value.get("dropout", 0.1)),
        )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Параметры воспроизводимого обучения."""

    seed: int = 42
    epochs: int = 5
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    bf16: bool = True
    num_workers: int = 0
    require_gpu: bool = True

    def __post_init__(self) -> None:
        """Проверяет численные параметры обучения."""
        positive = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "max_grad_norm": self.max_grad_norm,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Параметры должны быть положительными: {invalid}")
        if self.weight_decay < 0 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("Некорректные weight_decay или warmup_ratio")
        if self.num_workers < 0:
            raise ValueError("num_workers не может быть отрицательным")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrainingConfig:
        """Создаёт параметры обучения из YAML-объекта."""
        allowed = {
            "seed",
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "weight_decay",
            "warmup_ratio",
            "max_grad_norm",
            "bf16",
            "num_workers",
            "require_gpu",
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
    training: TrainingConfig
    output_root: str = "runs"
    primary_metric: str = "exact_micro_f1"

    def __post_init__(self) -> None:
        """Проверяет идентификатор и критерий выбора модели."""
        if not self.run_id or any(character.isspace() for character in self.run_id):
            raise ValueError("run_id должен быть непустым и не содержать пробелы")
        if not self.data_config or not self.output_root:
            raise ValueError("Нужны data_config и output_root")
        if self.primary_metric != "exact_micro_f1":
            raise ValueError("Best checkpoint выбирается только по exact_micro_f1")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExperimentConfig:
        """Создаёт полный эксперимент из YAML-объекта."""
        allowed = {
            "run_id",
            "data_config",
            "encoder",
            "model",
            "training",
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
            training=TrainingConfig.from_mapping(
                _require_mapping(value.get("training", {}), name="training")
            ),
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
