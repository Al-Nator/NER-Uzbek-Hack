"""Общие строгие проверки YAML-конфигураций."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """Проверяет, что значение является отображением."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} должен быть YAML-объектом")
    return value


def require_sequence(value: Any, *, name: str) -> Sequence[Any]:
    """Проверяет, что значение является YAML-массивом."""
    if not isinstance(value, list):
        raise TypeError(f"{name} должен быть YAML-массивом")
    return value


def check_keys(value: Mapping[str, Any], allowed: set[str], *, name: str) -> None:
    """Запрещает опечатки и неизвестные ключи конфигурации."""
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{name}: неизвестные ключи: {sorted(unknown)}")


def load_yaml(path: Path) -> Mapping[str, Any]:
    """Читает один YAML-файл с проверкой корневого типа."""
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return require_mapping(value, name=str(path))
