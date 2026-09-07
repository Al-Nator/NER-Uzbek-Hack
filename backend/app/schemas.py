"""HTTP-слой сервиса распознавания сущностей."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator


class Document(BaseModel):
    """Входной документ HTTP-запроса."""

    model_config = ConfigDict(strict=True)

    hash: str = Field(min_length=1, description="Unique identifier within the batch.")
    text: str = Field(description="Original text; no normalization is applied.")

    @field_validator("hash", "text")
    @classmethod
    def valid_unicode(cls, value: str) -> str:
        """Отклоняет строки с некорректными Unicode-символами."""
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("Field must contain valid Unicode text.") from exc
        return value


class PredictRequest(RootModel[list[Document]]):
    """Непустой пакет входных документов."""

    root: list[Document] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_hashes(self) -> "PredictRequest":
        """Проверяет уникальность идентификаторов документов."""
        hashes = [document.hash for document in self.root]
        if len(hashes) != len(set(hashes)):
            raise ValueError("Document hashes must be unique within the batch.")
        return self


class Entity(BaseModel):
    """Проверяемый HTTP-формат символьной сущности."""

    model_config = ConfigDict(strict=True, revalidate_instances="always")

    label: Literal["ORG", "NAME", "GEO"]
    start: int = Field(ge=0, description="Inclusive Python Unicode character index.")
    end: int = Field(gt=0, description="Exclusive Python Unicode character index.")

    @model_validator(mode="after")
    def valid_span(self) -> "Entity":
        """Проверяет непустой символьный интервал."""
        if self.end <= self.start:
            raise ValueError("Entity end must be greater than start.")
        return self


class Prediction(BaseModel):
    """Результат обработки одного документа."""

    hash: str
    entities: list[Entity]


class PredictResponse(BaseModel):
    """Полный ответ на пакетный запрос."""

    data: list[Prediction]


class HealthResponse(BaseModel):
    """Ответ служебного маршрута."""

    status: Literal["ok"] = "ok"


class ValidationIssue(BaseModel):
    """Ошибка проверки конкретного поля."""

    location: list[str | int]
    message: str
    type: str


class ErrorDetail(BaseModel):
    """Код, сообщение и подробности ошибки."""

    code: str
    message: str
    details: list[ValidationIssue] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Общая оболочка ответа об ошибке."""

    error: ErrorDetail
