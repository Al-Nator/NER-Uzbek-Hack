# Uzbek NER — FastAPI backend

HTTP-слой проекта для выделения `ORG`, `NAME` и `GEO` по контракту
[`ner_uz_hackathon_participant/API.md`](../ner_uz_hackathon_participant/API.md).

## Локальный запуск

Команды выполняются из каталога `backend/`: сначала `cd backend` из корня проекта.
Нужен Python 3.10+.
Backend использует отдельное окружение от окружения анализа данных.

```bash
uv venv .venv
uv pip sync --python .venv/bin/python requirements.txt
uv run --no-project --python .venv/bin/python uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Swagger UI: <http://localhost:8000/docs>. OpenAPI: <http://localhost:8000/openapi.json>.
В режиме разработки к команде Uvicorn можно добавить `--reload`.

## Docker

```bash
docker build -t ner-uz-solution .
docker run --rm -p 8000:8000 ner-uz-solution
```

Или:

```bash
docker compose up --build
```

Образ запускается от непривилегированного пользователя. Контекст сборки — только
каталог `backend/`; данные и исследовательские артефакты в образ не входят.
Docker healthcheck проверяет доступность HTTP-процесса через `/livez`.
Swagger UI загружает JS/CSS с CDN; API и OpenAPI JSON работают без интернета.

## API

| Метод и путь | Назначение |
| --- | --- |
| `GET /livez` | Проверка HTTP-процесса: `200 {"status":"ok"}` |
| `GET /healthz` | Готовность модели: `200 {"status":"ok"}`, без инференса |
| `POST /api/v1/predict` | Выделение сущностей в батче |
| `GET /docs` | Swagger UI |
| `GET /openapi.json` | Схема API |

Авторизация не требуется. Тело предсказания — непустой массив, а не объект с `data`:

```bash
curl -i http://localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  --data '[{"hash":"example-001","text":"Ali Toshkent shahrida ishlaydi."}]'
```

Пример формата ответа (конкретные сущности определяются моделью):

```json
{
  "data": [
    {
      "hash": "example-001",
      "entities": [
        {"label": "NAME", "start": 0, "end": 3},
        {"label": "GEO", "start": 4, "end": 12}
      ]
    }
  ]
}
```

Backend сохраняет порядок, `hash` и исходный текст. `hash` должен быть непустой
строкой и уникальным внутри батча. Пустой `text` допустим. Дополнительные входные
поля игнорируются. Координаты — индексы символов Unicode в Python, полуинтервал
`[start, end)`; это не байты и не UTF-16 индексы JavaScript. Во frontend для
выделения упоминания с эмодзи можно использовать `Array.from(text).slice(start, end).join("")`.

Ответ модели проверяется целиком: один набор сущностей на каждый текст, допустимые
метки, целочисленные границы в пределах исходного текста, отсутствие дублей.
При ошибке частичный результат не возвращается.

### Ошибки

Все обрабатываемые ошибки имеют общий JSON-формат:

```json
{
  "error": {
    "code": "model_unavailable",
    "message": "Model is not connected.",
    "details": []
  }
}
```

| Код HTTP | Причина |
| --- | --- |
| `400` | Невозможно прочитать тело запроса, например некорректная UTF-8 кодировка |
| `413` | Превышен настроенный лимит батча или длины текста |
| `422` | Невалидный JSON, пустой массив, повторяющиеся `hash`, неверные поля/типы |
| `500` | Ошибка модели или некорректные предсказания; подробности в серверном логе |
| `503` | Модель не подключена |

Ошибки валидации включают `details` с путём поля, сообщением и типом ошибки.
Тело запроса в ошибки валидации не копируется.

## Настройки

Для настройки скопируйте `.env.example` в `.env` или задайте переменные окружения.
Значения окружения имеют приоритет над `.env`. По умолчанию конфигурация не нужна.

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `NER_MAX_BATCH_SIZE` | `64` | Максимум документов в одном запросе |
| `NER_MAX_TEXT_LENGTH` | `50000` | Максимальная длина одного текста в символах Python |
| `NER_MAX_TOTAL_CHARACTERS` | `500000` | Максимальная сумма длин текстов в батче |
| `NER_CORS_ORIGINS` | `[]` | Разрешённые origin frontend, JSON-массив |

Тексты в пределах лимитов передаются обработчику полностью. Превышение лимита
даёт `413`, текст не обрезается. Лимиты проверяются после разбора JSON.
Например, для Vite можно задать `NER_CORS_ORIGINS=["http://localhost:5173"]`.

## Проверки

```bash
uv pip sync --python .venv/bin/python requirements-dev.txt
uv run --no-project --python .venv/bin/python pytest -q
uv run --no-project --python .venv/bin/python ruff check app tests
uv run --no-project --python .venv/bin/python ruff format --check app tests
```

Тесты проверяют HTTP-контракт,
Unicode-координаты, лимиты, обработку ошибок, жизненный цикл приложения,
сериализацию обработки запросов и доступность служебных маршрутов.

Версии runtime- и dev-зависимостей, включая транзитивные, зафиксированы в
`requirements.txt` и `requirements-dev.txt`. Исходные списки — `requirements.in`
и `requirements-dev.in`. Для обновления lock-файлов используется `uv pip compile`.
