# Удалённые эксперименты на A100

Историческая очередь **s24 → s25 → s30 → s31**:
[`A100_QUEUE.md`](A100_QUEUE.md). Defaults CLI ниже всё ещё относятся к s22/s23;
для новой очереди обязательно передавать её manifest, suffix и session.

## Оглавление

- [Метрики и веса](#сначала-метрики-затем-веса)
- [Схема](#схема)
- [Подготовка s22/s23](#подготовка-s22-и-s23)
- [Наблюдение](#наблюдение)
- [Возврат результатов](#возврат-результатов)
- [Время](#оценка-времени)
- [Текущий возврат s22/s23](#текущий-возврат-s22s23)

Команды и параметры **s30/s31** находятся в [серии 3](THIRD_SERIES.md).
Они используют отдельные manifest, suffix и zellij-session; запускать default
`start` ниже для серии 3 нельзя — его defaults относятся к s22/s23.

## Сначала метрики, затем веса

При медленной связи можно сразу импортировать результаты и всю MLflow-историю:

```bash
uv run python scripts/remote_experiments.py sync --metrics-only
uv run python scripts/remote_experiments.py sync
```

Первый вызов проверяет SHA-256 лёгких артефактов, импортирует метрики/params/tags
и создаёт `.checkpoint-transfer-pending` в локальном run-каталоге.
Второй использует partial rsync, догружает веса, проверяет полный manifest и
снимает маркер. Статус `complete` описывает обучение на источнике; наличие
локальных весов отдельно показывает MLflow tag `uzner.transfer.checkpoints`.
Веса в MLflow не загружаются. `logs/mlflow_run_id.txt` сохраняет ID источника,
`logs/mlflow_local_run_id.txt` содержит ID локальной копии. Повторный sync
не создаёт второй завершённый MLflow run для того же source ID.

## Схема

`scripts/remote_experiments.py` управляет хостом `alnator`, где проект
лежит в `/home/danya/NER-Uzbek-Hack`.

```text
локальный CLI ─SSH→ zellij на alnator ─→ обучение на A100
                         ├─ runs/ и checkpoints
                         └─ отдельный MLflow SQLite

готовый run ─rsync + SHA-256→ локальный runs/
            └─ params, tags, вся metric history и лёгкие артефакты
               импортируются в основной локальный MLflow
```

Обучение пишет напрямую в remote SQLite и не зависит от tunnel или
состояния локального компьютера. Checkpoint-ы в MLflow не дублируются.

## Подготовка `s22` и `s23`

Все команды ниже выполняются из локального корня репозитория.
Сначала нужно сверить Git commit, хэши train/dev, A100, собрать
remote uv-среду и сделать BF16 forward:

```bash
uv run --extra train python scripts/remote_experiments.py prepare --preflight
```

Эта же команда поднимает remote MLflow в отдельной zellij-сессии.

Полный запуск только `s22` и `s23` с отдельным suffix:

```bash
uv run --extra train python scripts/remote_experiments.py start
```

Для A100 используется отдельный манифест
`configs/series/second_a100.yaml`: train batch `8`, eval batch `16`, accumulation
`1`, BF16 и отключённый gradient checkpointing. Effective batch остаётся
равен `8`, как в основном протоколе.

По умолчанию будут созданы:

- `s22_xlmr_large_bioes_constrained_s2-sequence-a100-v2`;
- `s23_xlmr_large_bioes_crf_s2-sequence-a100-v2`.

Процесс идёт в detached zellij-сессии `uzner-s22-s23`. Выход из SSH
или закрытие tunnel не останавливает обучение.

## Наблюдение

Краткий статус GPU и panes:

```bash
uv run --extra train python scripts/remote_experiments.py status
```

Полный живой training log:

```bash
uv run --extra train python scripts/remote_experiments.py attach
```

В zellij отсоединиться, не останавливая run: `Ctrl+O`, затем `D`.

Для remote MLflow оставить в отдельном терминале:

```bash
uv run --extra train python scripts/remote_experiments.py tunnel
# http://127.0.0.1:5001
```

Основной локальный MLflow по-прежнему доступен на `http://127.0.0.1:5000`.

## Возврат результатов

Вариант с автоматическим rsync сразу после каждого завершённого run:

```bash
uv run --extra train python scripts/remote_experiments.py watch
```

Если локальный компьютер был выключен, позже выполнить идемпотентный
импорт всех готовых run-ов:

```bash
uv run --extra train python scripts/remote_experiments.py sync
```

`rsync` сначала пишет в `runs/.remote-incoming/<run_id>`, проверяет
размер и SHA-256 каждого файла, а затем атомарно переименовывает
каталог. Импорт MLflow сохраняет run name, params, tags, все точки
метрик с их step/timestamp и лёгкие артефакты. Теги
`uzner.transfer.source_run_id` и `uzner.transfer.source_host` связывают копию
с remote run-ом. Повторный `sync` не создаёт дубль.

## Оценка времени

На 512 train и 128 dev документах A100 smoke дал:

| Run | Train | Eval | Throughput | Peak VRAM |
|---|---:|---:|---:|---:|
| `s22` | 9.1 с | 2.3 с | 6.3–7.8k tokens/s | 10.98 GiB |
| `s23` | 30.8 с | 2.7 с | 2.0–2.1k tokens/s | 10.98 GiB |

По этому замеру ориентир для полных пяти эпох — `25–35` минут для
`s22` и `65–80` минут для `s23`, всего около `1.5–2 часов`.

## Текущий возврат s22/s23

Завершённые A100-v2 runs импортированы в основной MLflow. Проверены все
метрические точки (value/step/timestamp), params и SHA-256 лёгких файлов:
s22 — **9 949** точек, s23 — **15 189**, по **2 158** metric keys.
История не заменена одной последней точкой; source ID сохранены отдельно.

Фактическое обучение: s22 **23,6 мин**, s23 **90,4 мин**, без подготовки до
`run_started`. Лучшие exact micro-F1: **0.9010106** и **0.8982669** соответственно.

Догрузка тяжёлых checkpoint-ов запущена как отдельный user-service и переживает
завершение задачи Codex. Проверить её, не запуская второй rsync одновременно:

```bash
systemctl --user status uzner-return-a100
journalctl --user -u uzner-return-a100 -f
```

Пока присутствует `runs/<run_id>/.checkpoint-transfer-pending`, веса ещё не
подтверждены локально. После успеха service проверит полный manifest и снимет
маркер. При обрыве сети можно повторить `sync`: partial-файлы переиспользуются.
