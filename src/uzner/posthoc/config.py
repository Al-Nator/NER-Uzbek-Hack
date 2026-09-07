"""Явные параметры отдельных posthoc-абляций."""

from dataclasses import dataclass

from uzner.domain import LABELS


@dataclass(frozen=True)
class Variant:
    """Одна проверка без обучения; значения по умолчанию входят в resolved config."""

    name: str
    operation: str = "identity"
    base: str = "s62"
    labels: tuple[str, ...] = LABELS
    min_length: int = 4
    min_seeds: int = 1
    normalized: bool = False
    support: int = 3
    purity: float = 1.0
    propensity: float = 0.95
    boundary: str = "conservative"
    confirmer: str = "s45"
    threshold: float = 0.5
    org_threshold: float | None = None
    name_threshold: float | None = None
    selection: str = "greedy"
    repeat_after: bool = False
    boundary_after: bool = False

    def __post_init__(self) -> None:
        """Проверяет диапазоны и запрещает опечатки в операциях и названиях."""
        if not self.name or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in self.name
        ):
            raise ValueError("Некорректное имя варианта")
        operations = {
            "identity",
            "repeat",
            "boundary",
            "lex_add",
            "lex_relabel",
            "short",
            "confirm",
            "vote",
            "lex_confirm",
            "gp",
            "gp_vote",
        }
        if self.operation not in operations or self.selection not in {"greedy", "optimal"}:
            raise ValueError("Неизвестная операция или выбор spans")
        if not set(self.labels) <= set(LABELS) or not self.labels:
            raise ValueError("Неизвестные классы")
        if min(self.min_length, self.min_seeds, self.support) < 1:
            raise ValueError("Параметры количества должны быть положительными")
        for value in (
            self.purity,
            self.propensity,
            self.threshold,
            self.org_threshold,
            self.name_threshold,
        ):
            if value is not None and not 0 <= value <= 1:
                raise ValueError("Порог вне [0, 1]")
        if self.boundary not in {"right", "word", "trim", "conservative", "dictionary"}:
            raise ValueError("Неизвестное правило границ")
