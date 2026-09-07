"""HTTP-слой сервиса распознавания сущностей."""

from pydantic import TypeAdapter

from app.predictor import Predictor
from app.schemas import Document, Entity, Prediction, PredictResponse

ENTITY_BATCH = TypeAdapter(list[list[Entity]])


class ModelUnavailableError(RuntimeError):
    """Ошибка вызова сервиса без обработчика."""

    pass


class InvalidPredictionError(RuntimeError):
    """Ошибка целостности предсказаний модели."""

    pass


class PredictionService:
    """Проверка и упаковка результатов модели в HTTP-ответ."""

    def __init__(self, predictor: Predictor | None = None) -> None:
        """Инициализирует состояние обработчика."""
        self._predictor = predictor

    @property
    def ready(self) -> bool:
        """Возвращает признак готовности модели."""
        return self._predictor is not None

    def predict(self, documents: list[Document]) -> PredictResponse:
        """Возвращает сущности для документов с исходными символьными границами."""
        if self._predictor is None:
            raise ModelUnavailableError("Model is not connected.")

        # Проверяем весь результат до формирования HTTP-ответа.
        raw_entities = self._predictor.predict([document.text for document in documents])
        batches = ENTITY_BATCH.validate_python(
            [list(entities) for entities in raw_entities], strict=True, from_attributes=True
        )
        if len(batches) != len(documents):
            raise InvalidPredictionError("Model returned an incorrect number of results.")

        results = []
        for document, entities in zip(documents, batches, strict=True):
            seen: set[tuple[str, int, int]] = set()
            for entity in entities:
                if entity.end > len(document.text):
                    raise InvalidPredictionError("Entity exceeds the original text length.")
                key = (entity.label, entity.start, entity.end)
                if key in seen:
                    raise InvalidPredictionError("Model returned a duplicate entity.")
                seen.add(key)
            results.append(Prediction(hash=document.hash, entities=entities))
        return PredictResponse(data=results)

    def close(self) -> None:
        """Освобождает ресурсы обработчика."""
        predictor, self._predictor = self._predictor, None
        if predictor is not None:
            predictor.close()
