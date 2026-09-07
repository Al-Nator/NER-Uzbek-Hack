"""HTTP-слой сервиса распознавания сущностей."""

import logging
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings
from app.predictor import Predictor, PredictorFactory
from app.schemas import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ValidationIssue,
)
from app.service import ModelUnavailableError, PredictionService

logger = logging.getLogger(__name__)


def error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    details: list[ValidationIssue] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Формирует единый JSON-ответ об ошибке."""
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, details=details or []))
    return JSONResponse(status_code=status_code, content=body.model_dump(), headers=headers)


def create_app(
    *,
    settings: Settings | None = None,
    predictor_factory: PredictorFactory | None = None,
) -> FastAPI:
    """Создаёт HTTP-приложение с необязательной фабрикой модели."""
    config = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Сериализуем инференс, сохраняя доступность служебных маршрутов.
        """Управляет загрузкой и освобождением ресурсов модели."""
        app.state.inference_limiter = anyio.CapacityLimiter(1)
        if predictor_factory is not None:
            predictor = await anyio.to_thread.run_sync(predictor_factory)
            if not isinstance(predictor, Predictor):
                raise TypeError("predictor_factory must return a Predictor.")
            app.state.service = PredictionService(predictor)
            logger.info("NER model loaded; inference is ready.")
        else:
            logger.info("NER API started without a model; inference readiness returns 503.")
        try:
            yield
        finally:
            await anyio.to_thread.run_sync(app.state.service.close)

    app = FastAPI(
        title="Uzbek NER API",
        version="1.0.0",
        description=(
            "Batch extraction of ORG, NAME and GEO entities. Offsets refer to the original "
            "text and use Python Unicode character indices: text[start:end]."
        ),
        lifespan=lifespan,
    )
    app.state.service = PredictionService()
    app.state.settings = config

    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Преобразует ошибки входных данных в ответ 422."""
        return error_response(
            422,
            "validation_error",
            "Invalid request body.",
            details=[
                ValidationIssue(
                    location=list(error["loc"]), message=error["msg"], type=error["type"]
                )
                for error in exc.errors()
            ],
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Приводит ошибки HTTP к общей схеме."""
        codes = {
            404: "not_found",
            405: "method_not_allowed",
            413: "payload_too_large",
            500: "inference_error",
            503: "model_unavailable",
        }
        return error_response(
            exc.status_code,
            codes.get(exc.status_code, "http_error"),
            str(exc.detail),
            headers=exc.headers,
        )

    @app.get("/livez", response_model=HealthResponse, tags=["Health"])
    async def liveness() -> HealthResponse:
        """Проверяет доступность HTTP-процесса."""
        return HealthResponse()

    @app.get(
        "/healthz",
        response_model=HealthResponse,
        responses={503: {"model": ErrorResponse}},
        tags=["Health"],
    )
    async def health() -> HealthResponse:
        """Проверяет готовность модели без инференса."""
        if not app.state.service.ready:
            raise HTTPException(status_code=503, detail="Model is not connected.")
        return HealthResponse()

    @app.post(
        "/api/v1/predict",
        response_model=PredictResponse,
        responses={
            413: {
                "model": ErrorResponse,
                "description": "Configured batch or text limit exceeded.",
            },
            422: {"model": ErrorResponse, "description": "Malformed or invalid request."},
            500: {
                "model": ErrorResponse,
                "description": "Inference failed or returned invalid spans.",
            },
            503: {"model": ErrorResponse, "description": "Model is not connected."},
        },
        tags=["NER"],
    )
    async def predict(payload: PredictRequest) -> PredictResponse:
        """Возвращает сущности для документов с исходными символьными границами."""
        documents = payload.root
        if len(documents) > config.max_batch_size:
            raise HTTPException(413, f"Batch exceeds {config.max_batch_size} documents.")
        if any(len(document.text) > config.max_text_length for document in documents):
            raise HTTPException(413, f"Text exceeds {config.max_text_length} characters.")
        if sum(len(document.text) for document in documents) > config.max_total_characters:
            raise HTTPException(413, f"Batch exceeds {config.max_total_characters} characters.")

        if not app.state.service.ready:
            raise HTTPException(503, "Model is not connected.")
        try:
            return await anyio.to_thread.run_sync(
                app.state.service.predict,
                documents,
                limiter=app.state.inference_limiter,
            )
        except ModelUnavailableError as exc:
            raise HTTPException(503, "Model is not connected.") from exc
        except Exception as exc:
            logger.exception("NER inference failed for a batch of %d documents.", len(documents))
            raise HTTPException(500, "Prediction failed.") from exc

    return app


# Подключение модели: create_app(predictor_factory=YourPredictor).
# Жизненный цикл, сериализация вызовов и проверка spans остаются в HTTP-слое.
app = create_app()
