# Журнал экспериментов

Фактические результаты добавляются после появления полного набора артефактов.

| Run ID | Encoder | Tags | Head | Decoder | Seed | Micro-F1 | Статус |
|---|---|---|---|---|---:|---:|---|
| `b00_official` | DistilmBERT | BIO | softmax | greedy | 42 | — | подготовлен конфиг |
| `b01_reference` | DistilmBERT | BIO | softmax | greedy | 42 | — | подготовлен конфиг |
| `e10_xlmr_base_bio` | XLM-R-base | BIO | softmax | greedy | 42 | — | подготовлен конфиг |
| `e11_mdeberta_v3_base_bio` | mDeBERTa-v3-base | BIO | softmax | greedy | 42 | — | подготовлен конфиг |
| `e12_mmbert_base_bio` | mmBERT-base | BIO | softmax | greedy | 42 | — | подготовлен конфиг |
| `a20_xlmr_base_bio_constrained` | XLM-R-base | BIO | softmax | constrained | 42 | — | подготовлен конфиг |
| `a21_xlmr_base_bioes_constrained` | XLM-R-base | BIOES | softmax | constrained | 42 | — | подготовлен конфиг |
| `a22_xlmr_base_bio_crf` | XLM-R-base | BIO | CRF | CRF | 42 | — | подготовлен конфиг |
| `a23_xlmr_base_bioes_crf` | XLM-R-base | BIOES | CRF | CRF | 42 | — | подготовлен конфиг |
