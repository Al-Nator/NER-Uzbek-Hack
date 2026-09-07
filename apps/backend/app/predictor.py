"""Контракт адаптера модели без импорта ML-библиотек и загрузки весов."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Protocol


class EntitySpan(Protocol):
    """Символьный span, совместимый с общей dataclass uzner.domain.Entity."""

    @property
    def label(self) -> str:
        """Возвращает метку ORG, NAME или GEO."""
        ...

    @property
    def start(self) -> int:
        """Возвращает включённую левую границу исходной Unicode-строки."""
        ...

    @property
    def end(self) -> int:
        """Возвращает исключённую правую границу исходной Unicode-строки."""
        ...


class Predictor(ABC):
    """Родительский класс модели, передаваемой в create_app через фабрику."""

    @abstractmethod
    def predict(self, texts: list[str]) -> Sequence[Sequence[EntitySpan]]:
        """Возвращает spans для каждого текста, сохраняя порядок документов.

        Поддерживается общий тип uzner.domain.Entity. Границы относятся к
        неизменённому тексту; нормализация и перестановка документов запрещены.
        Ресурсы загружаются в конструкторе подкласса один раз при старте.
        HTTP-слой сериализует вызовы и выполняет их в рабочем потоке.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Освобождает ресурсы модели при завершении приложения."""
        return None


PredictorFactory = Callable[[], Predictor]
