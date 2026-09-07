# Приложения

Frontend/backend перенесены из upstream-коммита
[`51a176f`](https://github.com/Al-Nator/NER-Uzbek-Hack/commit/51a176f) в `apps/`.
Исследовательский код остаётся в `src/uzner/`; reference организаторов не изменён.
Upstream объединён с релизом сервиса; приложения находятся только в `apps/`.
Исходный коммит сохранён в истории Git. README/Compose используют новые пути
и производственный ансамбль, а не каркас без модели.

- `frontend/` — React/TypeScript, подсветка Unicode spans, JSONL-ввод и экспорт.
- `backend/` — FastAPI-контракт, валидация, readiness и обработка ошибок.
- `backend/app/ensemble.py` — адаптер настоящего s62+c02 к HTTP-интерфейсу.

Полный GPU-сервис: **из корня репозитория** `docker compose up --build`.
UI — порт 3000, API — 8000. [Развёртывание и измерения](../docs/SERVING.md).

`app.main:app` — изолированный HTTP-каркас без модели: `/healthz` возвращает 503.
`app.ensemble:app` — производственная точка входа с тремя моделями.
Не используйте отдельный `backend/Dockerfile` как финальную модельную посылку.
