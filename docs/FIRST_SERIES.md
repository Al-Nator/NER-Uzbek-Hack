# Первая серия

Манифест запуска: `configs/series/first.yaml`. В этой серии нет
подбора гиперпараметров: задача — сначала получить интерпретируемую
точку отсчёта.

## Порядок

1. **Tokenizer preflight.** До обучения снять representability, subword
   fragmentation и число окон для каждого encoder-а.
2. **`baseline`.** `b00_official` запускает неизменённый код
   организаторов и выбирает checkpoint по dev loss. `b01_reference` проверяет
   новый контур на том же DistilmBERT/BIO/softmax.
3. **`encoders`.** XLM-R-base, mDeBERTa-v3-base и mmBERT-base сравниваются
   при одинаковых BIO, greedy softmax, context 512, seed 42 и effective batch 8.
4. **`decoding`.** На стабильной XLM-R-base проверяются constrained BIO,
   BIOES, CRF и взаимодействие BIOES + CRF.

После серии лучший механизм переносится на лучший encoder. Это уже
следующая серия, а не скрытая донастройка текущей.

## Результат

Исправленная серия `s1-offsetfix-mlflow-r2` завершена полностью. Лучший encoder —
mDeBERTa-v3-base с exact micro-F1 `0.8804`. Лучший decoding на XLM-R-base —
BIOES + CRF с `0.8756`; он даёт `+0.0206` к XLM-R BIO/softmax/greedy. Во второй
серии BIOES + CRF и более быстрый BIOES constrained переносятся на mDeBERTa.
Полная таблица и точные выводы находятся в [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Что считается успешным run-ом

- `status.json` содержит `complete`;
- есть predictions, exact/slice/error metrics, tokenizer audit и report;
- есть resolved config, data hashes, environment и artifact manifest;
- для нового контура есть `best` и `last`; official baseline сохраняет только
  выдаваемый им `best` и не поддерживает resume.

## Команды

```bash
# проверить порядок, ничего не обучая
uv run python scripts/run_series.py --stage all --dry-run

# запускать этапы последовательно на GPU
uv run python scripts/run_series.py --stage baseline
uv run python scripts/run_series.py --stage encoders
uv run python scripts/run_series.py --stage decoding
```

Существующие run-каталоги не перезаписываются. Канонический завершённый прогон
имеет suffix `s1-offsetfix-mlflow-r2`. Для нового повторения используйте другой
общий suffix, например `--run-suffix s1-reseed-01`.
