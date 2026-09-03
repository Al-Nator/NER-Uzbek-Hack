"""Типизированные описания расширяемых источников данных."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from uzner.config_helpers import check_keys, require_mapping, require_sequence

Split = Literal["train", "dev", "test"]
SourceKind = Literal["gold", "synthetic", "pseudo"]


@dataclass(frozen=True, slots=True)
class DataSourceConfig:
    """Один подключаемый источник документов."""

    name: str
    path: str
    split: Split
    kind: SourceKind = "gold"
    weight: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        """Проверяет имя, тип и вес источника."""
        if not self.name or not self.path:
            raise ValueError("У источника обязательны name и path")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError(f"Некорректный split: {self.split!r}")
        if self.kind not in {"gold", "synthetic", "pseudo"}:
            raise ValueError(f"Некорректный kind: {self.kind!r}")
        if self.weight <= 0:
            raise ValueError("weight источника должен быть положительным")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DataSourceConfig:
        """Создаёт описание источника из YAML-объекта."""
        allowed = {"name", "path", "split", "kind", "weight", "enabled"}
        check_keys(value, allowed, name="data.sources[]")
        return cls(
            name=value.get("name"),
            path=value.get("path"),
            split=value.get("split"),
            kind=value.get("kind", "gold"),
            weight=float(value.get("weight", 1.0)),
            enabled=bool(value.get("enabled", True)),
        )


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Набор источников и параметры окон токенизации."""

    sources: tuple[DataSourceConfig, ...]
    max_length: int = 256
    stride: int = 64

    def __post_init__(self) -> None:
        """Проверяет уникальность источников и параметры окон."""
        names = [source.name for source in self.sources]
        if len(set(names)) != len(names):
            raise ValueError("Имена источников данных должны быть уникальными")
        enabled_splits = {source.split for source in self.sources if source.enabled}
        if not {"train", "dev"}.issubset(enabled_splits):
            raise ValueError("Нужен хотя бы один включённый train и dev источник")
        if self.max_length < 8:
            raise ValueError("max_length должен быть не меньше 8")
        if not 0 <= self.stride < self.max_length:
            raise ValueError("stride должен удовлетворять 0 <= stride < max_length")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DataConfig:
        """Создаёт конфигурацию данных из YAML-объекта."""
        check_keys(value, {"sources", "max_length", "stride"}, name="data")
        raw_sources = require_sequence(value.get("sources"), name="data.sources")
        return cls(
            sources=tuple(
                DataSourceConfig.from_mapping(require_mapping(item, name=f"data.sources[{index}]"))
                for index, item in enumerate(raw_sources)
            ),
            max_length=int(value.get("max_length", 256)),
            stride=int(value.get("stride", 64)),
        )


def resolve_sources(
    data: DataConfig,
    *,
    split: Split,
    project_root: Path,
) -> tuple[tuple[str, Path], ...]:
    """Разрешает включённые пути заданного split относительно корня."""
    return tuple(
        (source.name, (project_root / source.path).resolve())
        for source in data.sources
        if source.enabled and source.split == split
    )
