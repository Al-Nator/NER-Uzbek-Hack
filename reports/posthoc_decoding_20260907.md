# Posthoc: все 71 прогона на original dev

Дата: 07.09.2026. 1 500 документов; exact `(hash, label, start, end)`. Без обучения.
Δ — относительно s62 + точные повторы (0.915247653). Отбор по dev; это не независимая оценка.
Public: s62 0.8861; s62 + повторы 0.8897; s48 0.8778 — сообщения пользователя.

[Выводы и ограничения](../docs/POSTHOC_DECODING.md) · [MLflow](http://127.0.0.1:5000/#/experiments/7/runs)

| Вариант / полный run | Операция | F1 | Δ F1 | P | R | TP / FP / FN | Половина A / B |
|---|---|---:|---:|---:|---:|---|---|
| [c02_norm_lex_then_repeat](../runs/posthoc_20260907_combinations_v1_c02_norm_lex_then_repeat/metrics/dev.json) | lex_add | 0.917448 | +0.002200 | 0.913139 | 0.921798 | 7096 / 675 / 602 | 0.912428 / 0.922613 |
| [c04_lex_confirm_then_repeat](../runs/posthoc_20260907_combinations_v1_c04_lex_confirm_then_repeat/metrics/dev.json) | lex_confirm | 0.916171 | +0.000923 | 0.912394 | 0.919979 | 7082 / 680 / 616 | 0.910320 / 0.922189 |
| [c10_repeat_then_lex_confirm](../runs/posthoc_20260907_combinations_v1_c10_repeat_then_lex_confirm/metrics/dev.json) | lex_confirm | 0.916171 | +0.000923 | 0.912394 | 0.919979 | 7082 / 680 / 616 | 0.910320 / 0.922189 |
| [c08_repeat_then_norm_lex](../runs/posthoc_20260907_combinations_v1_c08_repeat_then_norm_lex/metrics/dev.json) | lex_add | 0.916155 | +0.000907 | 0.911852 | 0.920499 | 7086 / 685 / 612 | 0.909879 / 0.922613 |
| [c11_repeat_then_s45](../runs/posthoc_20260907_combinations_v1_c11_repeat_then_s45/metrics/dev.json) | confirm | 0.916115 | +0.000867 | 0.906731 | 0.925695 | 7126 / 733 / 572 | 0.910495 / 0.921912 |
| [c03_strict_lex_then_repeat](../runs/posthoc_20260907_combinations_v1_c03_strict_lex_then_repeat/metrics/dev.json) | lex_add | 0.916101 | +0.000853 | 0.912382 | 0.919849 | 7081 / 680 / 617 | 0.910204 / 0.922168 |
| [c09_repeat_then_strict_lex](../runs/posthoc_20260907_combinations_v1_c09_repeat_then_strict_lex/metrics/dev.json) | lex_add | 0.916101 | +0.000853 | 0.912382 | 0.919849 | 7081 / 680 / 617 | 0.910204 / 0.922168 |
| [c05_s45_confirm_then_repeat](../runs/posthoc_20260907_combinations_v1_c05_s45_confirm_then_repeat/metrics/dev.json) | confirm | 0.915816 | +0.000568 | 0.905524 | 0.926345 | 7131 / 744 / 567 | 0.909620 / 0.922214 |
| [c07_gp020_then_repeat](../runs/posthoc_20260907_combinations_v1_c07_gp020_then_repeat/metrics/dev.json) | gp_vote | 0.915580 | +0.000333 | 0.909826 | 0.921408 | 7093 / 703 / 605 | 0.908651 / 0.922695 |
| [c06_s45_name_then_repeat](../runs/posthoc_20260907_combinations_v1_c06_s45_name_then_repeat/metrics/dev.json) | confirm | 0.915563 | +0.000315 | 0.910680 | 0.920499 | 7086 / 695 / 612 | 0.908141 / 0.923198 |
| [c12_repeat_then_s45_name](../runs/posthoc_20260907_combinations_v1_c12_repeat_then_s45_name/metrics/dev.json) | confirm | 0.915482 | +0.000234 | 0.910774 | 0.920239 | 7084 / 694 / 614 | 0.908257 / 0.922916 |
| [c01_exact_repeat_name_org](../runs/posthoc_20260907_combinations_v1_c01_exact_repeat_name_org/metrics/dev.json) | repeat | 0.915325 | +0.000077 | 0.915087 | 0.915562 | 7048 / 654 / 650 | 0.908414 / 0.922390 |
| [d19_repeat_name_org](../runs/posthoc_20260907_v1_d19_repeat_name_org/metrics/dev.json) | repeat | 0.915265 | +0.000018 | 0.914968 | 0.915562 | 7048 / 655 / 650 | 0.908414 / 0.922269 |
| [d11_repeat_2](../runs/posthoc_20260907_v1_d11_repeat_2/metrics/dev.json) | repeat | 0.915259 | +0.000011 | 0.912247 | 0.918290 | 7069 / 680 / 629 | 0.908673 / 0.922027 |
| [c13_gp_name020_then_repeat](../runs/posthoc_20260907_combinations_v1_c13_gp_name020_then_repeat/metrics/dev.json) | gp_vote | 0.915254 | +0.000007 | 0.911598 | 0.918940 | 7074 / 686 / 624 | 0.908557 / 0.922129 |
| [d10_repeat_4](../runs/posthoc_20260907_v1_d10_repeat_4/metrics/dev.json) | repeat | 0.915248 | +0.000000 | 0.912353 | 0.918161 | 7068 / 679 / 630 | 0.908650 / 0.922027 |
| [c00_repeat_control](../runs/posthoc_20260907_combinations_v1_c00_repeat_control/metrics/dev.json) | repeat | 0.915248 | +0.000000 | 0.912353 | 0.918161 | 7068 / 679 / 630 | 0.908650 / 0.922027 |
| [d50_confirm_s45](../runs/posthoc_20260907_v1_d50_confirm_s45/metrics/dev.json) | confirm | 0.914766 | -0.000482 | 0.910120 | 0.919460 | 7078 / 699 / 620 | 0.910919 / 0.918734 |
| [d17_repeat_normalized](../runs/posthoc_20260907_v1_d17_repeat_normalized/metrics/dev.json) | repeat | 0.914485 | -0.000762 | 0.910073 | 0.918940 | 7074 / 699 / 624 | 0.907700 / 0.921463 |
| [d14_repeat_name](../runs/posthoc_20260907_v1_d14_repeat_name/metrics/dev.json) | repeat | 0.914423 | -0.000824 | 0.915495 | 0.913354 | 7031 / 649 / 667 | 0.908250 / 0.920748 |
| [d33_lex_norm_3_090](../runs/posthoc_20260907_v1_d33_lex_norm_3_090/metrics/dev.json) | lex_add | 0.914308 | -0.000940 | 0.915917 | 0.912705 | 7026 / 645 / 672 | 0.909814 / 0.918933 |
| [d18_repeat_two_seeds](../runs/posthoc_20260907_v1_d18_repeat_two_seeds/metrics/dev.json) | repeat | 0.914234 | -0.001014 | 0.915245 | 0.913224 | 7030 / 651 / 668 | 0.908484 / 0.920137 |
| [d52_confirm_s45_name](../runs/posthoc_20260907_v1_d52_confirm_s45_name/metrics/dev.json) | confirm | 0.913914 | -0.001333 | 0.914866 | 0.912964 | 7028 / 654 / 670 | 0.908182 / 0.919810 |
| [d12_repeat_6](../runs/posthoc_20260907_v1_d12_repeat_6/metrics/dev.json) | repeat | 0.913826 | -0.001422 | 0.913648 | 0.914004 | 7036 / 665 / 662 | 0.907156 / 0.920668 |
| [d55_lex_confirm](../runs/posthoc_20260907_v1_d55_lex_confirm/metrics/dev.json) | lex_confirm | 0.913616 | -0.001632 | 0.916362 | 0.910886 | 7012 / 640 / 686 | 0.908857 / 0.918505 |
| [d32_lex_raw_5_100](../runs/posthoc_20260907_v1_d32_lex_raw_5_100/metrics/dev.json) | lex_add | 0.913545 | -0.001703 | 0.916351 | 0.910756 | 7011 / 640 / 687 | 0.908740 / 0.918483 |
| [d31_lex_raw_3_095](../runs/posthoc_20260907_v1_d31_lex_raw_3_095/metrics/dev.json) | lex_add | 0.913485 | -0.001762 | 0.916231 | 0.910756 | 7011 / 641 / 687 | 0.908624 / 0.918483 |
| [d56_lex_confirm_norm](../runs/posthoc_20260907_v1_d56_lex_confirm_norm/metrics/dev.json) | lex_confirm | 0.913344 | -0.001904 | 0.916209 | 0.910496 | 7009 / 641 / 689 | 0.908764 / 0.918054 |
| [d72_vote_gp_020](../runs/posthoc_20260907_v1_d72_vote_gp_020/metrics/dev.json) | gp_vote | 0.913284 | -0.001964 | 0.913343 | 0.913224 | 7030 / 667 / 668 | 0.908369 / 0.918335 |
| [d37_lex_geo](../runs/posthoc_20260907_v1_d37_lex_geo/metrics/dev.json) | lex_add | 0.913262 | -0.001986 | 0.916307 | 0.910236 | 7007 / 640 / 691 | 0.908740 / 0.917911 |
| [d15_repeat_org](../runs/posthoc_20260907_v1_d15_repeat_org/metrics/dev.json) | repeat | 0.913247 | -0.002000 | 0.915752 | 0.910756 | 7011 / 645 / 687 | 0.907336 / 0.919304 |
| [d30_lex_raw_2_090](../runs/posthoc_20260907_v1_d30_lex_raw_2_090/metrics/dev.json) | lex_add | 0.913199 | -0.002048 | 0.915524 | 0.910886 | 7012 / 647 / 686 | 0.908157 / 0.918384 |
| [d54_confirm_s45_geo](../runs/posthoc_20260907_v1_d54_confirm_s45_geo/metrics/dev.json) | confirm | 0.913193 | -0.002054 | 0.914204 | 0.912185 | 7022 / 659 / 676 | 0.909371 / 0.917129 |
| [d71_vote_gp_010](../runs/posthoc_20260907_v1_d71_vote_gp_010/metrics/dev.json) | gp_vote | 0.913052 | -0.002196 | 0.911456 | 0.914653 | 7041 / 684 / 657 | 0.908812 / 0.917412 |
| [d75_vote_gp_name_020](../runs/posthoc_20260907_v1_d75_vote_gp_name_020/metrics/dev.json) | gp_vote | 0.912987 | -0.002261 | 0.915491 | 0.910496 | 7009 / 647 / 689 | 0.908086 / 0.918020 |
| [d13_repeat_8](../runs/posthoc_20260907_v1_d13_repeat_8/metrics/dev.json) | repeat | 0.912981 | -0.002266 | 0.914170 | 0.911795 | 7019 / 659 / 679 | 0.905869 / 0.920279 |
| [d51_confirm_s48](../runs/posthoc_20260907_v1_d51_confirm_s48/metrics/dev.json) | confirm | 0.912903 | -0.002344 | 0.906819 | 0.919070 | 7075 / 727 / 623 | 0.907429 / 0.918557 |
| [d35_lex_name](../runs/posthoc_20260907_v1_d35_lex_name/metrics/dev.json) | lex_add | 0.912683 | -0.002564 | 0.916328 | 0.909067 | 6998 / 639 / 700 | 0.907171 / 0.918340 |
| [d73_vote_gp_035](../runs/posthoc_20260907_v1_d73_vote_gp_035/metrics/dev.json) | gp_vote | 0.912664 | -0.002583 | 0.914450 | 0.910886 | 7012 / 656 / 686 | 0.907807 / 0.917656 |
| [d34_lex_norm_5_100](../runs/posthoc_20260907_v1_d34_lex_norm_5_100/metrics/dev.json) | lex_add | 0.912505 | -0.002743 | 0.915969 | 0.909067 | 6998 / 642 / 700 | 0.906962 / 0.918197 |
| [d53_confirm_s45_org](../runs/posthoc_20260907_v1_d53_confirm_s45_org/metrics/dev.json) | confirm | 0.912472 | -0.002775 | 0.913542 | 0.911406 | 7016 / 664 / 682 | 0.907738 / 0.917337 |
| [d00_s62](../runs/posthoc_20260907_v1_d00_s62/metrics/dev.json) | identity | 0.912400 | -0.002848 | 0.916285 | 0.908548 | 6994 / 639 / 704 | 0.907171 / 0.917768 |
| [d22_trim_quotes](../runs/posthoc_20260907_v1_d22_trim_quotes/metrics/dev.json) | boundary | 0.912400 | -0.002848 | 0.916285 | 0.908548 | 6994 / 639 / 704 | 0.907171 / 0.917768 |
| [d24_dictionary_boundary](../runs/posthoc_20260907_v1_d24_dictionary_boundary/metrics/dev.json) | boundary | 0.912400 | -0.002848 | 0.916285 | 0.908548 | 6994 / 639 / 704 | 0.907171 / 0.917768 |
| [d36_lex_org](../runs/posthoc_20260907_v1_d36_lex_org/metrics/dev.json) | lex_add | 0.912340 | -0.002907 | 0.916165 | 0.908548 | 6994 / 640 / 704 | 0.907055 / 0.917768 |
| [d70_vote_gp_050](../runs/posthoc_20260907_v1_d70_vote_gp_050/metrics/dev.json) | gp_vote | 0.912340 | -0.002907 | 0.916165 | 0.908548 | 6994 / 640 / 704 | 0.907171 / 0.917647 |
| [d16_repeat_geo](../runs/posthoc_20260907_v1_d16_repeat_geo/metrics/dev.json) | repeat | 0.912331 | -0.002917 | 0.913519 | 0.911146 | 7014 / 664 / 684 | 0.907415 / 0.917405 |
| [d38_relabel_raw](../runs/posthoc_20260907_v1_d38_relabel_raw/metrics/dev.json) | lex_relabel | 0.912269 | -0.002978 | 0.916154 | 0.908418 | 6993 / 640 / 705 | 0.907429 / 0.917240 |
| [d76_vote_gp_org_020](../runs/posthoc_20260907_v1_d76_vote_gp_org_020/metrics/dev.json) | gp_vote | 0.912205 | -0.003042 | 0.914707 | 0.909717 | 7003 / 653 / 695 | 0.906941 / 0.917613 |
| [d40_short_unknown_2](../runs/posthoc_20260907_v1_d40_short_unknown_2/metrics/dev.json) | short | 0.912175 | -0.003072 | 0.916361 | 0.908028 | 6990 / 638 / 708 | 0.907007 / 0.917482 |
| [d74_vote_gp_065](../runs/posthoc_20260907_v1_d74_vote_gp_065/metrics/dev.json) | gp_vote | 0.912157 | -0.003091 | 0.917916 | 0.906469 | 6978 / 624 / 720 | 0.906911 / 0.917550 |
| [d20_right_word](../runs/posthoc_20260907_v1_d20_right_word/metrics/dev.json) | boundary | 0.912139 | -0.003109 | 0.916023 | 0.908288 | 6992 / 641 / 706 | 0.907429 / 0.916975 |
| [d39_relabel_norm](../runs/posthoc_20260907_v1_d39_relabel_norm/metrics/dev.json) | lex_relabel | 0.912139 | -0.003109 | 0.916023 | 0.908288 | 6992 / 641 / 706 | 0.907171 / 0.917240 |
| [c15_s45_repeat](../runs/posthoc_20260907_combinations_v1_c15_s45_repeat/metrics/dev.json) | repeat | 0.911995 | -0.003253 | 0.907944 | 0.916082 | 7052 / 715 / 646 | 0.908189 / 0.915932 |
| [d21_both_word](../runs/posthoc_20260907_v1_d21_both_word/metrics/dev.json) | boundary | 0.911226 | -0.004022 | 0.915105 | 0.907379 | 6985 / 648 / 713 | 0.907429 / 0.915124 |
| [d23_conservative_boundary](../runs/posthoc_20260907_v1_d23_conservative_boundary/metrics/dev.json) | boundary | 0.911226 | -0.004022 | 0.915105 | 0.907379 | 6985 / 648 / 713 | 0.907429 / 0.915124 |
| [d02_s45](../runs/posthoc_20260907_v1_d02_s45/metrics/dev.json) | identity | 0.909576 | -0.005672 | 0.910345 | 0.908807 | 6996 / 689 / 702 | 0.906766 / 0.912480 |
| [c14_s48_repeat](../runs/posthoc_20260907_combinations_v1_c14_s48_repeat/metrics/dev.json) | repeat | 0.909267 | -0.005981 | 0.902804 | 0.915822 | 7050 / 759 / 648 | 0.902653 / 0.916099 |
| [d41_short_unknown_3](../runs/posthoc_20260907_v1_d41_short_unknown_3/metrics/dev.json) | short | 0.908495 | -0.006753 | 0.916909 | 0.900234 | 6930 / 628 / 768 | 0.905358 / 0.911729 |
| [d66_gp_name_020](../runs/posthoc_20260907_v1_d66_gp_name_020/metrics/dev.json) | gp | 0.907522 | -0.007726 | 0.906757 | 0.908288 | 6992 / 719 / 706 | 0.901895 / 0.913301 |
| [d63_gp_035](../runs/posthoc_20260907_v1_d63_gp_035/metrics/dev.json) | gp | 0.907361 | -0.007887 | 0.904375 | 0.910366 | 7008 / 741 / 690 | 0.902146 / 0.912718 |
| [d62_gp_020](../runs/posthoc_20260907_v1_d62_gp_020/metrics/dev.json) | gp | 0.907266 | -0.007982 | 0.898992 | 0.915692 | 7049 / 792 / 649 | 0.903086 / 0.911558 |
| [d69_gp_optimal_020](../runs/posthoc_20260907_v1_d69_gp_optimal_020/metrics/dev.json) | gp | 0.907266 | -0.007982 | 0.898992 | 0.915692 | 7049 / 792 / 649 | 0.903086 / 0.911558 |
| [d60_gp_050](../runs/posthoc_20260907_v1_d60_gp_050/metrics/dev.json) | gp | 0.906712 | -0.008536 | 0.908782 | 0.904651 | 6964 / 699 / 734 | 0.900655 / 0.912929 |
| [d68_gp_optimal_050](../runs/posthoc_20260907_v1_d68_gp_optimal_050/metrics/dev.json) | gp | 0.906712 | -0.008536 | 0.908782 | 0.904651 | 6964 / 699 / 734 | 0.900655 / 0.912929 |
| [d01_s33](../runs/posthoc_20260907_v1_d01_s33/metrics/dev.json) | identity | 0.906700 | -0.008548 | 0.908889 | 0.904521 | 6963 / 698 / 735 | 0.900514 / 0.913049 |
| [d03_s48](../runs/posthoc_20260907_v1_d03_s48/metrics/dev.json) | identity | 0.906604 | -0.008643 | 0.904670 | 0.908548 | 6994 / 737 / 704 | 0.900178 / 0.913267 |
| [d61_gp_010](../runs/posthoc_20260907_v1_d61_gp_010/metrics/dev.json) | gp | 0.906240 | -0.009008 | 0.893150 | 0.919719 | 7080 / 847 / 618 | 0.902248 / 0.910341 |
| [d67_gp_org_020](../runs/posthoc_20260907_v1_d67_gp_org_020/metrics/dev.json) | gp | 0.905993 | -0.009255 | 0.903709 | 0.908288 | 6992 / 745 / 706 | 0.900857 / 0.911263 |
| [d64_gp_065](../runs/posthoc_20260907_v1_d64_gp_065/metrics/dev.json) | gp | 0.904996 | -0.010252 | 0.912343 | 0.897766 | 6911 / 664 / 787 | 0.898022 / 0.912155 |
| [d65_gp_080](../runs/posthoc_20260907_v1_d65_gp_080/metrics/dev.json) | gp | 0.902958 | -0.012290 | 0.916054 | 0.890231 | 6853 / 628 / 845 | 0.895169 / 0.910933 |

Конфиги: [51 исходная абляция и 4 контроля](../configs/posthoc/decoding_v1.yaml),
[16 составных/контрольных вариантов](../configs/posthoc/combinations_v1.yaml).
Полные resolved config, source snapshot, hashes, детальные срезы, ошибки и предсказания — в каждом run.
Одинаковый F1 не всегда означает одинаковые предсказания. Отрицательные и нулевые результаты сохранены.
У d14–d19 исходной матрицы используется токенный поиск (пробелы объединяются).
c01 отдельно проверяет строго строковый поиск только NAME/ORG. c00 в точности повторяет d10.
