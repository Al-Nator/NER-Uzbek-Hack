"""Mechanically replace only notebook section 5 with its reproducible audit."""
from pathlib import Path
from textwrap import dedent
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eda.gazetteer_analysis import OUT


def cell(kind, cell_id, source):
    result = dict(cell_type=kind, id=cell_id, metadata={},
                  source=(dedent(source).strip() + "\n").splitlines(keepends=True))
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    base, combined = summary["coverage"][0], summary["coverage"][-1]
    cells = [
        cell("markdown", "74106926", '''
        ## 5. Источники gazetteer

        Gazetteer — словарь названий с типом сущности и ссылкой на реальный объект.
        Проверены **GeoNames, OpenStreetMap, Wikidata и два официальных источника организаций Узбекистана**.
        Для каждого ниже указаны формат, лицензия, охват, ограничения и польза. Дата проверки: **05.09.2026 (Москва)**.

        Практическая часть: скачан полный GeoNames `UZ.zip`, построен словарь GEO и измерено
        дополнительное покрытие `dev` относительно `train`. Остальные источники исследованы по документации
        и страницам доступа: их объём и покрытие NER-корпуса не измерялись.
        Все таблицы и исходный snapshot хранятся в `artifacts/gazetteer_analysis/`.
        '''),
        cell("code", "gazetteer-source-catalogue", '''
        from pathlib import Path
        import json
        import pandas as pd
        from IPython.display import display
        from scripts.eda.gazetteer_analysis import OUT as GAZETTEER_DIR, SOURCES as GAZETTEER_SOURCES

        gazetteer_sources_df = pd.DataFrame(GAZETTEER_SOURCES)
        display(gazetteer_sources_df[
            ["source", "labels", "format", "license", "coverage", "limits", "benefit", "status"]
        ].style.hide(axis="index").set_properties(**{"text-align": "left"}))
        '''),
        cell("markdown", "gazetteer-source-details", '''
        ### 5.1. Где брать данные и какие поля извлекать

        **GeoNames → GEO.** [UZ.zip](https://download.geonames.org/export/dump/UZ.zip),
        [описание схемы и лицензии](https://download.geonames.org/export/dump/readme.txt).
        В UTF-8 TSV сохранять `geonameid`, `name`, `asciiname`, `alternatenames`,
        `feature_class/code`, страну, административную иерархию и дату изменения.
        Взяты классы `A/P` (административные единицы/населённые пункты), отдельно расширение
        `H/T/V/L` (вода/рельеф/растительность/территории); `S/R` исключены из этого прототипа.
        `alternatenames` не содержит языковых и исторических флагов. При их необходимости
        использовать [alternateNamesV2](https://download.geonames.org/export/dump/alternateNamesV2.zip)
        с фильтром по UZ geonameid. У самого gazetteer-дампа сейчас **CC BY 4.0**;
        версия лицензии взята из скачанного README. `UZ.zip` не покрывает иностранные места в узбекских текстах.

        **OpenStreetMap → GEO.** [Выгрузка Узбекистана Geofabrik](https://download.geofabrik.de/asia/uzbekistan.html)
        доступна в PBF, SHP и GeoPackage. Для полного набора тегов предпочесть PBF.
        Обрабатывать `place`, `boundary=administrative`, `waterway`, `natural` и именованные улицы,
        согласовав границы GEO с нашей разметкой. Извлекать `name`, доступные `name:uz/name:ru`,
        `official_name`, `short_name`, `alt_name`; `old_name` хранить отдельно.
        [Семантика имён](https://wiki.openstreetmap.org/wiki/Key:name) важна для разделителей и вариантов.
        Дедуплицировать по типу объекта + ID, затем проверять совпадающие node/way/relation.
        `brand` и `operator` дают кандидатов ORG, но название здания или магазина нельзя автоматически
        считать организацией. [Данные OSM — ODbL](https://www.openstreetmap.org/copyright):
        учитывать attribution и условия распространения производных баз. Для полной выборки использовать extract;
        [публичный Nominatim запрещает систематическое скачивание списков объектов](https://operations.osmfoundation.org/policies/nominatim/).

        **Wikidata → NAME/ORG/GEO.** [Query Service](https://query.wikidata.org/),
        [JSON/API/dumps](https://www.wikidata.org/wiki/Wikidata:Data_access),
        [CC0 для структурированных данных](https://www.wikidata.org/wiki/Wikidata:Licensing).
        Хранить QID, label и каждый alias отдельной строкой с языком, revision и происхождением.
        Тип определять по `P31/P279`, связь с Узбекистаном — по гражданству `P27`, стране `P17`
        или расположению штаб-квартиры `P159/P17`. `uz`, `uz-cyrl`, `ru`, `en` запрашиваются по наличию;
        язык и письменность — разные поля. Фильтр гражданства исключает часть исторических/зарубежных лиц,
        а отсутствие `P17` не означает, что организация не узбекская. Популярные люди представлены лучше
        обычных участников комментариев: нельзя обещать полное покрытие NAME.

        **КТЯДР → ORG.** [Реестр](https://registr.stat.uz/) и
        [официальный регламент](https://stat.uz/img/xizmatlar/reglamentinteraktiv-xizmatlar.pdf)
        подтверждают сервис сведений по STIR (ИНН). На проверке корень сайта перенаправляет в OneID.
        Публичный bulk-export и лицензия повторного использования не подтверждены:
        это источник проверки названий, а не готовая выгрузка всех компаний.
        Сохранять юридическое имя, STIR, статус и дату; бренды и сокращения сверять отдельно.
        Статистические таблицы с количеством предприятий не являются словарями названий.

        **ЦБ Узбекистана → ORG.** [Реестр коммерческих банков](https://cbu.uz/uz/credit-organizations/banks/head-offices/)
        содержит HTML-карточки и ссылку на XLSX от **03.09.2026**; есть языковые версии сайта.
        Это точечный источник официальных имён банков, а не всех организаций страны.
        Для сопоставления версий использовать стабильный идентификатор/номер лицензии, а не адрес.
        [Другие финансовые реестры](https://cbu.uz/uz/credit-organizations/) расширяют отраслевой охват.
        Наличие файла в открытом доступе не устанавливает открытую лицензию на его перераспространение;
        условия для этого XLSX в просмотренной карточке не подтверждены.
        '''),
        cell("code", "gazetteer-wikidata-queries", '''
        # Шаблоны для первого просмотра, не результаты выгрузки.
        # Каждый запрос возвращает <=50 QID. Затем labels/aliases получаются API-батчем <=50 ID.
        WIKIDATA_PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        """
        wikidata_filters = {
            "NAME": "?item wdt:P31 wd:Q5; wdt:P27 wd:Q265 .",
            "ORG": """{ ?item wdt:P17 wd:Q265 . }
              UNION { ?item wdt:P159/wdt:P17 wd:Q265 . }
              ?item wdt:P31/wdt:P279* wd:Q43229 .""",
            "GEO": """?item wdt:P17 wd:Q265 .
              VALUES ?root { wd:Q486972 wd:Q56061 }
              ?item wdt:P31/wdt:P279* ?root .""",
        }
        gazetteer_queries = {
            label: WIKIDATA_PREFIXES + "SELECT DISTINCT ?item WHERE {\\n"
            + pattern + "\\n} ORDER BY ?item LIMIT 50\\n"
            for label, pattern in wikidata_filters.items()
        }
        query_dir = GAZETTEER_DIR / "queries"
        query_dir.mkdir(exist_ok=True)
        for label, query in gazetteer_queries.items():
            (query_dir / f"wikidata_{label.lower()}.rq").write_text(query, encoding="utf-8")
            print(f"{label}: {query_dir / f'wikidata_{label.lower()}.rq'}")

        display(pd.DataFrame([
            {"class": "NAME", "scope": "human (Q5), Uzbekistan citizenship (Q265)", "status": "template; not executed"},
            {"class": "ORG", "scope": "organization (Q43229); country or HQ country UZ", "status": "template; not executed"},
            {"class": "GEO", "scope": "human settlement (Q486972) or administrative entity (Q56061), UZ", "status": "template; not executed"},
        ]))
        '''),
        cell("markdown", "gazetteer-schema", '''
        ### 5.2. Формат словаря и правила использования

        Одна строка — один вариант имени одного объекта. Общие поля:
        `label, entity_id, name, normalized, language, script, alias_kind, source, license, snapshot/revision`.
        Для географии полезны страна/иерархия/тип объекта; для ORG — официальный идентификатор и статус.
        Одинаковое имя может вести к нескольким ID и классам: сохранять множество кандидатов.

        Ключ сравнения: `NFKC → casefold → унификация апострофов → схлопывание пробелов`.
        Исходное `name` сохраняется; текст документов и offsets не изменяются. Язык неизвестного alias
        не угадывается по алфавиту. В данной проверке нет транслитерации, stemming и автоматического
        удаления `-da/-ga/-dan/-ning`: для морфологии нужен отдельный эксперимент с контролем ложных совпадений.
        Алиасы короче трёх символов и без букв исключены заранее.

        Gazetteer использовать как признак для кандидатов модели: тип словаря, совпадение полного имени,
        число подходящих ID, источник и длина совпадения. На этапе поиска в тексте нужны границы слов,
        несколько длин совпадения и отображение нормализованных позиций на исходные offsets.
        Этот раздел измеряет только совпадение **уже размеченных** полных упоминаний; matcher и модель здесь не обучаются.

        Для Wikidata шаблоны ограничены первыми 50 ID и не являются оценкой объёма базы.
        Для следующих страниц использовать keyset-пагинацию по последнему ID; результат не усекать молча.
        Labels/aliases извлекать отдельным запросом к API с сохранением языков и revision.
        Большие выгрузки делать через dumps, учитывая
        [ограничения сервиса](https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/query_limits).
        GEO-шаблон охватывает поселения и административные единицы, а не всю природную географию.
        '''),
        cell("code", "gazetteer-geonames-audit", '''
        from scripts.eda.gazetteer_analysis import run as run_gazetteer_audit

        # Полностью локальный пересчёт из сохранённого UZ.zip; сеть не используется.
        gazetteer_summary = run_gazetteer_audit(download=False)
        display(pd.DataFrame([{
            "objects_in_UZ_zip": gazetteer_summary["objects"],
            "selected_objects": gazetteer_summary["selected_objects"],
            "alias_rows": gazetteer_summary["aliases"],
            "normalized_keys": gazetteer_summary["normalized_aliases"],
            "ambiguous_keys": gazetteer_summary["ambiguous_alias_keys"],
            "snapshot_sha256": gazetteer_summary["snapshot"]["sha256"],
        }]))
        display(pd.DataFrame(gazetteer_summary["feature_classes"].items(), columns=["feature_class", "objects"]))
        display(pd.DataFrame(gazetteer_summary["script_alias_rows"].items(), columns=["script", "alias_rows"]))
        gazetteer_coverage_df = pd.DataFrame(gazetteer_summary["coverage"])
        display(gazetteer_coverage_df.style.hide(axis="index"))
        '''),
        cell("markdown", "gazetteer-coverage-findings", f'''
        ### 5.3. Результат на текущем dev

        В `UZ.zip` **{summary['objects']:,} объектов**; после отбора типов и имен —
        **{summary['normalized_aliases']:,} нормализованных ключей** для {summary['selected_objects']:,} объектов.
        У {summary['ambiguous_alias_keys']:,} ключей несколько geonameid.
        `script` в таблице различает наличие латиницы/кириллицы, `other` включает остальные письменности;
        строки aliases не равны числу уникальных объектов или языков.

        Словарь GEO из `train` покрывает **{base['matched_mentions']} из {base['dev_mentions']} упоминаний dev
        ({base['mention_coverage_pct']}%)**. Добавление GeoNames даёт
        **{combined['matched_mentions']} ({combined['mention_coverage_pct']}%)**:
        дополнительно **{combined['added_mentions_vs_train']} упоминание и {combined['added_unique_vs_train']} уникальных форм**.
        Уникальное покрытие растёт с {base['unique_coverage_pct']}% до {combined['unique_coverage_pct']}%.
        GeoNames отдельно покрывает 464 упоминания (17,06%).

        Все дополнительные совпадения уже достигаются с `A/P`; расширение `H/T/V/L` не дало новых
        совпадений на этом dev. Это диагностический результат, а не основание удалять классы
        по проверочной выборке. Небольшой прирост согласуется с ограничением одной страной,
        окончаниями и вариантами записи; доля каждого фактора отдельно не измерялась.

        Это **лексическое покрытие gold-упоминаний**, а не precision/recall/F1 NER или проверка
        тождества географических объектов. Словарь создан независимо от dev; список новых форм ниже
        предназначен для просмотра и не добавляется обратно в train. Snapshot современный:
        для временного benchmark потребуется выгрузка не позднее границы времени теста.
        '''),
        cell("code", "gazetteer-review-tables", '''
        gazetteer_review_df = pd.read_csv(GAZETTEER_DIR / "new_dev_geo_review.csv")
        gazetteer_collisions_df = pd.read_csv(GAZETTEER_DIR / "cross_label_collisions.csv")
        display(gazetteer_review_df.head(50).style.hide(axis="index"))
        display(gazetteer_collisions_df.style.hide(axis="index"))
        '''),
        cell("markdown", "gazetteer-conclusion", '''
        ### Вывод по пункту 5

        Начать с GeoNames как дополнительного GEO-признака, затем проверить Wikidata для многоязычных
        NAME/ORG и известных мест. OSM полезен для локальных названий; официальные реестры — для
        проверки юридических имён и отраслевых словарей. Измеренная польза GeoNames умеренная,
        а польза остальных источников пока является гипотезой.

        Уже есть межклассовые коллизии: GEO-ключи совпадают с **6 NAME и 42 ORG упоминаниями dev**.
        Например, `Alisher Navoiy` может относиться к человеку или названному в его честь месту,
        `Paxtakor/Bunyodkor/Andijon` — к организации либо месту. Это показывает необходимость контекста;
        автоматическое присвоение GEO по одному совпадению создаёт ошибки.

        **Где посмотреть самому:** откройте таблицы выше или
        `artifacts/gazetteer_analysis/new_dev_geo_review.csv` (все 25 новых форм) и
        `cross_label_collisions.csv`. Для объекта GeoNames откройте `https://www.geonames.org/<id>/`.
        В `geonames_geo_aliases.csv` можно фильтровать `script=cyrillic/mixed` и неоднозначные имена.
        Перед внесением ручных решений скопируйте review-таблицу: автоматический пересчёт пересоздаёт отчёт.
        Онлайн-точки просмотра: [GeoNames](https://www.geonames.org/search.html?q=&country=UZ),
        [OSM](https://www.openstreetmap.org/), [Wikidata](https://query.wikidata.org/),
        [ЦБ](https://cbu.uz/uz/credit-organizations/banks/head-offices/), [КТЯДР](https://registr.stat.uz/).

        **Повторить расчёт:** `uv run python scripts/eda/gazetteer_analysis.py`.
        Для первоначальной загрузки при отсутствии snapshot: добавить `--download`.
        Присоединять dev-формы к словарю запрещено методикой этого эксперимента; оценивать
        NER-эффект затем на фиксированной проверочной выборке и отдельном итоговом test.
        '''),
    ]
    path = ROOT / "data_analysis.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    original = copy.deepcopy(notebook["cells"])
    start = next(i for i, c in enumerate(original) if c.get("id") == "74106926")
    end = next(i for i, c in enumerate(original) if c.get("id") == "1f7472be")
    notebook["cells"][start:end] = cells
    assert notebook["cells"][:start] == original[:start]
    assert notebook["cells"][start + len(cells):] == original[end:]
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Section 5: {len(cells)} cells; other sections preserved")


if __name__ == "__main__":
    build()
