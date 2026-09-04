"""Section 5: source catalogue and reproducible GeoNames lexical coverage.

Run with --download once; later runs use the verified local snapshot offline.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "gazetteer_analysis"
GN_URL = "https://download.geonames.org/export/dump/UZ.zip"
GN_README = "https://download.geonames.org/export/dump/readme.txt"
APOSTROPHES = str.maketrans({c: "'" for c in "’‘ʻʼʹ՚`´＇ꞌ"})
GEO_CLASSES = frozenset("APHTVL")
FIELDS = ("geonameid name asciiname alternatenames latitude longitude feature_class "
          "feature_code country_code cc2 admin1 admin2 admin3 admin4 population "
          "elevation dem timezone modification_date").split()
SOURCES = [
    dict(source="GeoNames UZ", labels="GEO", format="ZIP → UTF-8 TSV, 19 columns",
         license="CC BY 4.0 (gazetteer dump README)",
         coverage="Узбекистан: населённые пункты, административные и природные объекты; многоязычные aliases",
         limits="Языки/историчность aliases отсутствуют в UZ.txt; для них нужен alternateNamesV2; вне-UZ география не покрывается",
         benefit="Первый словарь GEO; фактическое покрытие измерено ниже",
         status="UZ.zip скачан, хеш зафиксирован; полный локальный аудит",
         url=GN_URL, documentation=GN_README),
    dict(source="OpenStreetMap / Geofabrik UZ", labels="GEO; ORG после проверки типа",
         format="OSM PBF; также SHP/GeoPackage; теги объектов",
         license="ODbL 1.0; attribution, условия для производных БД",
         coverage="Населённые пункты, районы, улицы, локальные названия и POI; полнота зависит от картирования",
         limits="Здание/бренд/оператор не равны GEO; дубли node/way/relation; name:uz/name:ru могут отсутствовать",
         benefit="Дополнение локальной географии, отсутствующей в GeoNames",
         status="Формат и доступ проверены; PBF не загружался, покрытие не измерено",
         url="https://download.geofabrik.de/asia/uzbekistan.html",
         documentation="https://www.openstreetmap.org/copyright"),
    dict(source="Wikidata", labels="NAME / ORG / GEO",
         format="Entity JSON/RDF; SPARQL JSON/CSV/TSV; labels + aliases + claims",
         license="CC0 для структурированных данных",
         coverage="Известные люди, организации и места; uz/uz-cyrl/ru/en формы по наличию",
         limits="Неполные labels и claims; QID не NER-класс; гражданство не равно языку; обычные люди представлены слабо",
         benefit="Многоязычные aliases и устойчивый QID; основной внешний кандидат для NAME",
         status="Документация проверена; запросы подготовлены, полной выгрузки нет",
         url="https://query.wikidata.org/",
         documentation="https://www.wikidata.org/wiki/Wikidata:Licensing"),
    dict(source="КТЯДР / registr.stat.uz", labels="ORG",
         format="Веб-сервис/сведения по STIR (ИНН); bulk-файл/API не подтверждён",
         license="Открытая лицензия массового повторного использования не подтверждена",
         coverage="Юридические лица Узбекистана в государственном регистре",
         limits="При проверке корень сайта перенаправляет в OneID; поиск по ID не даёт полного словаря; юридическое название отличается от бренда",
         benefit="Проверка официального названия и статуса известной организации",
         status="Регламент и сайт проверены; вход и массовая выгрузка не выполнялись",
         url="https://registr.stat.uz/",
         documentation="https://stat.uz/img/xizmatlar/reglamentinteraktiv-xizmatlar.pdf"),
    dict(source="ЦБ Узбекистана: реестр банков", labels="ORG",
         format="HTML-карточки + XLSX; языковые версии uz/oz/ru/en",
         license="Открытая лицензия на переиспользование XLSX не подтверждена",
         coverage="Коммерческие банки; другие финансовые реестры доступны отдельно",
         limits="Только финансовый сектор; XLSX и HTML могут иметь разные даты; краткие бренды требуют сверки",
         benefit="Небольшой проверяемый отраслевой справочник ORG",
         status="Подтверждена XLSX-ссылка от 03.09.2026; строки файла и покрытие не измерены",
         url="https://cbu.uz/uz/credit-organizations/banks/head-offices/",
         documentation="https://cbu.uz/uz/credit-organizations/"),
]


def normalize(text):
    """Normalize lookup keys, never the text holding annotation offsets."""
    text = unicodedata.normalize("NFKC", text).casefold().translate(APOSTROPHES)
    return re.sub(r"\s+", " ", text).strip()


def script_of(text):
    scripts = {s for c in text for s in ("LATIN", "CYRILLIC") if s in unicodedata.name(c, "")}
    return "mixed" if len(scripts) == 2 else next(iter(scripts), "other").lower()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def acquire_snapshot(download):
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path, metadata = raw / "UZ.zip", raw / "UZ.snapshot.json"
    if not path.exists():
        if not download:
            raise FileNotFoundError("Нет snapshot. Выполните python gazetteer_analysis.py --download")
        request = Request(GN_URL, headers={"User-Agent": "ai_hack-gazetteer-research/1.0"})
        with urlopen(request, timeout=45) as response:
            data = response.read(10_000_001)
            if len(data) > 10_000_000:
                raise ValueError("Unexpected GeoNames UZ snapshot size")
            with ZipFile(io.BytesIO(data)) as archive:
                if "UZ.txt" not in archive.namelist() or archive.testzip() is not None:
                    raise ValueError("Invalid GeoNames UZ archive")
            manifest = dict(url=GN_URL, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                            retrieved_at=datetime.now(timezone.utc).isoformat(),
                            last_modified=response.headers.get("Last-Modified"),
                            license="CC BY 4.0", license_source=GN_README)
        path.write_bytes(data)
        save_json(metadata, manifest)
    manifest = json.loads(metadata.read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("GeoNames snapshot hash mismatch")
    if download and not (raw / "geonames_readme.txt").exists():
        with urlopen(Request(GN_README, headers={"User-Agent": "ai_hack-gazetteer-research/1.0"}), timeout=45) as response:
            (raw / "geonames_readme.txt").write_bytes(response.read())
    return path, manifest


def load_geonames(path):
    with ZipFile(path) as archive:
        with archive.open("UZ.txt") as f:
            rows = list(csv.reader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t"))
    if any(len(row) != len(FIELDS) for row in rows):
        raise ValueError("Unexpected GeoNames columns")
    result = [dict(zip(FIELDS, row)) for row in rows]
    assert len({r["geonameid"] for r in result}) == len(result)
    assert all(r["country_code"] == "UZ" for r in result)
    return result


def build_aliases(records, revision):
    aliases = []
    for row in records:
        if row["feature_class"] not in GEO_CLASSES:
            continue
        seen = set()
        names = [("name", row["name"]), ("asciiname", row["asciiname"])]
        names += [("alternatenames", name) for name in row["alternatenames"].split(",")]
        for kind, name in names:
            key = normalize(name)
            if len(key) < 3 or not any(c.isalpha() for c in key) or name in seen:
                continue
            seen.add(name)
            aliases.append(dict(label="GEO", entity_id="geonames:" + row["geonameid"],
                                name=name, normalized=key, language="unknown", script=script_of(name),
                                alias_kind=kind, feature_class=row["feature_class"],
                                feature_code=row["feature_code"], country="UZ", source="GeoNames",
                                license="CC BY 4.0", snapshot_sha256=revision))
    return aliases


def read_mentions(path):
    mentions = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for entity in row["entities"]:
                mentions[entity["label"]].append(row["text"][entity["start"]:entity["end"]])
    return mentions


def measure_coverage(aliases):
    train = read_mentions(ROOT / "ner_uz_hackathon_participant/data/train.jsonl")
    dev = read_mentions(ROOT / "ner_uz_hackathon_participant/data/dev.jsonl")
    base = {normalize(m) for m in train["GEO"]}
    mentions = [normalize(m) for m in dev["GEO"]]
    unique = set(mentions)
    small = {a["normalized"] for a in aliases if a["feature_class"] in {"A", "P"}}
    broad = {a["normalized"] for a in aliases}
    strategies = {"train GEO": base, "GeoNames A/P only": small, "GeoNames A/P/H/T/V/L": broad,
                  "train + GeoNames A/P": base | small, "train + GeoNames A/P/H/T/V/L": base | broad}
    metrics = []
    for strategy, keys in strategies.items():
        matched, unique_matched = sum(m in keys for m in mentions), len(unique & keys)
        metrics.append(dict(strategy=strategy, alias_keys=len(keys), dev_mentions=len(mentions),
                            matched_mentions=matched, mention_coverage_pct=round(100 * matched / len(mentions), 2),
                            dev_unique=len(unique), matched_unique=unique_matched,
                            unique_coverage_pct=round(100 * unique_matched / len(unique), 2),
                            added_mentions_vs_train=sum(m in keys and m not in base for m in mentions),
                            added_unique_vs_train=len((unique & keys) - base)))
    ids = defaultdict(set)
    for alias in aliases:
        ids[alias["normalized"]].add(alias["entity_id"])
    examples = [dict(mention=m, dev_count=n, candidate_entities=len(ids[m]),
                     geonames_ids=";".join(sorted(ids[m])), manual_decision="pending")
                for m, n in Counter(mentions).most_common() if m in broad and m not in base][:50]
    collision_rows = []
    for split, data in (("train", train), ("dev", dev)):
        for label in ("NAME", "ORG"):
            values = [normalize(m) for m in data[label]]
            hits = [m for m in values if m in broad]
            collision_rows.append(dict(split=split, gold_label=label, mentions=len(values),
                                       matching_GEO_alias_mentions=len(hits),
                                       matching_GEO_alias_unique=len(set(hits)),
                                       examples="; ".join(m for m, _ in Counter(hits).most_common(10))))
    return metrics, examples, collision_rows


def run(download=False):
    OUT.mkdir(parents=True, exist_ok=True)
    path, snapshot = acquire_snapshot(download)
    records = load_geonames(path)
    aliases = build_aliases(records, snapshot["sha256"])
    metrics, examples, collisions = measure_coverage(aliases)
    save_csv(OUT / "sources.csv", SOURCES)
    save_csv(OUT / "geonames_geo_aliases.csv", aliases)
    save_csv(OUT / "coverage.csv", metrics)
    save_csv(OUT / "new_dev_geo_review.csv", examples,
             ["mention", "dev_count", "candidate_entities", "geonames_ids", "manual_decision"])
    save_csv(OUT / "cross_label_collisions.csv", collisions)
    key_ids = defaultdict(set)
    for alias in aliases:
        key_ids[alias["normalized"]].add(alias["entity_id"])
    summary = dict(snapshot=snapshot, objects=len(records),
                   feature_classes=dict(sorted(Counter(r["feature_class"] for r in records).items())),
                   selected_feature_classes=sorted(GEO_CLASSES),
                   selected_objects=len({a["entity_id"] for a in aliases}),
                   aliases=len(aliases), normalized_aliases=len(key_ids),
                   ambiguous_alias_keys=sum(len(v) > 1 for v in key_ids.values()),
                   script_alias_rows=dict(sorted(Counter(a["script"] for a in aliases).items())),
                   normalization="NFKC + casefold + apostrophe unification + whitespace collapse; no transliteration/stemming",
                   selection="A/P/H/T/V/L; alias >=3 code points with a letter; selected before looking at dev",
                   input_sha256={split: hashlib.sha256((ROOT / f"ner_uz_hackathon_participant/data/{split}.jsonl").read_bytes()).hexdigest()
                                 for split in ("train", "dev")},
                   coverage=metrics, caveat="Gold mention surface coverage; not NER precision, recall or F1. No model trained.")
    save_json(OUT / "summary.json", summary)
    save_json(OUT / "sources.json", SOURCES)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download missing GeoNames snapshot only")
    args = parser.parse_args()
    print(json.dumps(run(args.download), ensure_ascii=False, indent=2))
