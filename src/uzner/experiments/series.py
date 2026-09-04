"""Типизированный манифест последовательности экспериментов."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uzner.config_helpers import check_keys, load_yaml, require_mapping, require_sequence


@dataclass(frozen=True, slots=True)
class SeriesStage:
    """Один смысловой этап серии с упорядоченными конфигами."""

    name: str
    configs: tuple[str, ...]

    def __post_init__(self) -> None:
        """Проверяет имя и непустой список конфигов."""
        if not self.name or not self.configs:
            raise ValueError("Этапу нужны name и хотя бы один config")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SeriesStage:
        """Создаёт этап из YAML-объекта."""
        check_keys(value, {"name", "configs"}, name="series.stages[]")
        raw_configs = require_sequence(value.get("configs"), name="series.stages[].configs")
        return cls(str(value.get("name", "")), tuple(str(item) for item in raw_configs))


@dataclass(frozen=True, slots=True)
class SeriesConfig:
    """Полная серия в порядке запуска."""

    name: str
    stages: tuple[SeriesStage, ...]

    def __post_init__(self) -> None:
        """Запрещает пустые и повторяющиеся этапы и конфиги."""
        names = [stage.name for stage in self.stages]
        configs = [config for stage in self.stages for config in stage.configs]
        if not self.name or not self.stages:
            raise ValueError("Серии нужны name и stages")
        if len(names) != len(set(names)) or len(configs) != len(set(configs)):
            raise ValueError("Этапы и experiment configs не должны повторяться")

    def select(self, stage: str) -> tuple[str, ...]:
        """Возвращает всю серию или один именованный этап."""
        if stage == "all":
            return tuple(config for item in self.stages for config in item.configs)
        for item in self.stages:
            if item.name == stage:
                return item.configs
        raise ValueError(f"Неизвестный stage={stage!r}; доступны {names(self.stages)}")


def names(stages: tuple[SeriesStage, ...]) -> tuple[str, ...]:
    """Возвращает имена этапов для сообщения об ошибке."""
    return tuple(stage.name for stage in stages)


def load_series(path: Path) -> SeriesConfig:
    """Загружает и строго проверяет YAML-манифест серии."""
    root = load_yaml(path)
    check_keys(root, {"series"}, name=str(path))
    value = require_mapping(root.get("series"), name="series")
    check_keys(value, {"name", "stages"}, name="series")
    raw_stages = require_sequence(value.get("stages"), name="series.stages")
    return SeriesConfig(
        name=str(value.get("name", "")),
        stages=tuple(
            SeriesStage.from_mapping(require_mapping(item, name="series.stages[]"))
            for item in raw_stages
        ),
    )
