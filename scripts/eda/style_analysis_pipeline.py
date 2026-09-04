#!/usr/bin/env python3
"""Reproducible orchestration helpers for section 2 of data_analysis.ipynb.

The script never modifies the source train/dev JSONL files.  It prepares
stratified samples and worker batches, validates agent output, and merges the
independent annotations into auditable reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ner_uz_hackathon_participant" / "data"
ARTIFACT_DIR = ROOT / "artifacts" / "style_analysis"
BATCH_DIR = ARTIFACT_DIR / "batches"
ANNOTATION_DIR = ARTIFACT_DIR / "annotations"
SEED = 42
BATCH_SIZE = 50
TAXONOMY_VERSION = "2.0.0"
RUN_MODEL = "gpt-5.6-sol"
RUN_REASONING_EFFORT = "low"

GENRES = [
    "news_report",
    "official_announcement",
    "social_post",
    "comment_reply",
    "customer_review",
    "business_response",
    "advertisement",
    "classified_listing",
    "informational_educational",
    "quiz_poll",
    "reference_result",
    "personal_message",
    "mixed_composite",
    "other",
]

TOPICS = [
    "politics_government",
    "international_relations_conflict",
    "law_crime_security",
    "business_economy_finance",
    "construction_real_estate_urbanism",
    "technology_telecom",
    "transport_automotive_aviation",
    "sports_esports",
    "entertainment_music_fandom",
    "consumer_goods_services",
    "health_medicine",
    "education_science",
    "employment_migration",
    "society_social_issues",
    "religion_spirituality",
    "culture_history_language",
    "environment_agriculture",
    "travel_geography",
    "personal_daily_life",
    "other",
]

TEXT_FORMS = ["native_text", "speech_transcript", "image_ocr"]
REGISTERS = ["formal", "neutral", "informal", "mixed"]

GENRE_DEFINITIONS = {
    "news_report": "Third-person factual coverage of an event: a news story, report, review, or digest. A social-channel origin does not override this genre.",
    "official_announcement": "First-party statement by a government body, institution, company, or press service in its own voice; not third-person reporting about that body.",
    "social_post": "Standalone social-feed/channel post or caption that does not meet a more specific genre below. Treat it as a fallback, not a label for every text sourced from social media.",
    "comment_reply": "Short user reaction, opinion, conversational answer, or context-dependent fragment. Prefer this over other for ordinary readable comments.",
    "customer_review": "Customer evaluation of a product, place, course, or service, even when the platform is not named.",
    "business_response": "An organisation's reply to a customer, including templated replies.",
    "advertisement": "Promotion, sale announcement, or broad call to buy/order.",
    "classified_listing": "Structured private offer with price, contact details, or item parameters.",
    "informational_educational": "Explanation, instruction, advice, or teaching material.",
    "quiz_poll": "Explicit quiz, test, poll, or audience question that requests an answer/vote. Never use merely because a text is short, imperative, or contains a rhetorical question.",
    "reference_result": "Non-narrative schedule, result, profile, list, or media metadata.",
    "personal_message": "Greeting, invitation, or personal announcement.",
    "mixed_composite": "One record clearly concatenates texts of different genres or authors.",
    "other": "Unreadable/minimal material with insufficient communicative function, or no suitable genre. Do not use it as a fallback for a normal comment.",
}

TOPIC_DEFINITIONS = {
    "politics_government": "Domestic politics, elections, and public administration.",
    "international_relations_conflict": "Foreign affairs, wars, and international conflicts.",
    "law_crime_security": "Law, courts, crime, corruption, and security.",
    "business_economy_finance": "Business, economy, markets, banking, and finance.",
    "construction_real_estate_urbanism": "Real estate, construction, and urban infrastructure.",
    "technology_telecom": "IT, digital services, communications, and telecom.",
    "transport_automotive_aviation": "Cars, roads, public transport, rail, and aviation.",
    "sports_esports": "Sports, competitions, and esports.",
    "entertainment_music_fandom": "Film, music, shows, celebrities, and fandoms.",
    "consumer_goods_services": "Food, restaurants, shops, products, and household services.",
    "health_medicine": "Health, medicine, and pharmaceuticals.",
    "education_science": "Education, learning, and science.",
    "employment_migration": "Jobs, employment, and migration.",
    "society_social_issues": "Community life and social issues.",
    "religion_spirituality": "Religion and spiritual practices.",
    "culture_history_language": "Culture, history, literature, and language.",
    "environment_agriculture": "Environment, weather, natural resources, and agriculture.",
    "travel_geography": "Tourism, countries, cities, and landmarks.",
    "personal_daily_life": "Relationships, family, and everyday life.",
    "other": "The topic is indeterminate or outside the taxonomy.",
}

TEXT_FORM_DEFINITIONS = {
    "native_text": "Originally written/typed text, or no observable evidence of OCR/transcription.",
    "speech_transcript": "Speech transcribed from audio/video, often with speaker labels.",
    "image_ocr": "Text extracted from an image, often introduced by an OCR marker.",
}

REGISTER_DEFINITIONS = {
    "formal": "Official, institutional, professional, or deliberately formal language.",
    "neutral": "Unmarked informational language without a strongly formal or conversational tone.",
    "informal": "Conversational, colloquial, emotional, or slang-heavy language.",
    "mixed": "Substantial mixture of formal and informal registers in one record.",
}

OUTPUT_FIELDS = {
    "item_id",
    "genre",
    "topics",
    "text_form",
    "register",
    "evidence",
    "genre_confidence",
    "topics_confidence",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def entity_signature(record: dict[str, Any]) -> str:
    labels = sorted({entity["label"] for entity in record["entities"]})
    return "+".join(labels) if labels else "EMPTY"


def detect_script(text: str, mixed_threshold: float = 0.10) -> str:
    latin = 0
    cyrillic = 0
    for character in text:
        name = unicodedata.name(character, "")
        latin += "LATIN" in name
        cyrillic += "CYRILLIC" in name
    alphabetic = latin + cyrillic
    if alphabetic == 0:
        return "other"
    if min(latin, cyrillic) / alphabetic >= mixed_threshold:
        return "mixed"
    return "latin" if latin > cyrillic else "cyrillic"


def entity_count_bin(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 5:
        return "2-5"
    return "6+"


def quantile_thresholds(values: list[int], quantiles: tuple[float, ...]) -> list[int]:
    ordered = sorted(values)
    return [ordered[round((len(ordered) - 1) * quantile)] for quantile in quantiles]


def quantile_bin(value: int, thresholds: list[int]) -> str:
    return f"Q{1 + sum(value > threshold for threshold in thresholds)}"


def normalise_template(text: str) -> str:
    value = text.casefold()
    value = re.sub(r"https?://\S+|www\.\S+", "<url>", value)
    value = re.sub(r"@[\w.]+", "<user>", value)
    value = re.sub(r"\+?\d[\d\s()\-]{5,}\d", "<phone>", value)
    value = re.sub(r"\b\d+(?:[.,:/-]\d+)*\b", "<num>", value)
    value = re.sub(r"[^\w<>]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def stratum_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item["split"],
        item["entity_signature"],
        item["script"],
        item["length_quantile"],
        item["entity_count_bin"],
        item["dated_hash"],
        item["repeat_kind"],
    )


def serialise_stratum(key: tuple[Any, ...]) -> dict[str, Any]:
    names = [
        "split",
        "entity_signature",
        "script",
        "length_quantile",
        "entity_count_bin",
        "dated_hash",
        "repeat_kind",
    ]
    return dict(zip(names, key))


def apportion_with_rare_quotas(
    groups: dict[tuple[Any, ...], list[dict[str, Any]]],
    target: int,
    seed: int,
    rare_quota_fraction: float,
) -> dict[tuple[Any, ...], int]:
    """Allocate a fixed sample and reserve a small, explicit rare-strata quota."""
    if target > sum(map(len, groups.values())):
        raise ValueError("sample target is larger than the population")

    allocation = {key: 0 for key in groups}
    quota_budget = min(round(target * rare_quota_fraction), len(groups), target)
    candidates = [
        key
        for key, items in groups.items()
        if len(items) <= 5 or key[-1] != "unique"
    ]
    candidates.sort(
        key=lambda key: (
            key[-1] == "unique",
            len(groups[key]),
            stable_fraction(repr(key), seed),
        )
    )
    for key in candidates[:quota_budget]:
        allocation[key] = 1

    remaining = target - sum(allocation.values())
    while remaining:
        capacity = {
            key: len(items) - allocation[key]
            for key, items in groups.items()
            if len(items) > allocation[key]
        }
        if not capacity:
            raise RuntimeError("allocation exhausted before reaching target")
        total_capacity = sum(capacity.values())
        exact = {key: remaining * size / total_capacity for key, size in capacity.items()}
        additions = {
            key: min(size, math.floor(exact[key])) for key, size in capacity.items()
        }
        added = sum(additions.values())
        for key, count in additions.items():
            allocation[key] += count
        remaining -= added
        if not remaining:
            break
        order = sorted(
            capacity,
            key=lambda key: (
                exact[key] - math.floor(exact[key]),
                capacity[key],
                stable_fraction(repr(key), seed + 1),
            ),
            reverse=True,
        )
        progressed = False
        for key in order:
            if remaining == 0:
                break
            if allocation[key] < len(groups[key]):
                allocation[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("failed to finish allocation")
    return allocation


def stratified_sample(
    population: list[dict[str, Any]],
    target: int,
    seed: int,
    rare_quota_fraction: float,
) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], int]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in population:
        groups[stratum_key(item)].append(item)
    allocation = apportion_with_rare_quotas(groups, target, seed, rare_quota_fraction)
    sample = []
    for key, items in groups.items():
        ordered = sorted(
            items,
            key=lambda item: stable_fraction(item["hash"], seed + 2),
        )
        sample.extend(ordered[: allocation[key]])
    sample.sort(key=lambda item: stable_fraction(item["hash"], seed + 3))
    if len(sample) != target:
        raise AssertionError(f"expected {target} sampled items, got {len(sample)}")
    return sample, allocation


def build_population() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_records: list[tuple[str, dict[str, Any]]] = []
    sources = {}
    for split in ("train", "dev"):
        path = DATA_DIR / f"{split}.jsonl"
        records = read_jsonl(path)
        sources[split] = {
            "path": str(path.relative_to(ROOT)),
            "documents": len(records),
            "sha256": sha256_file(path),
        }
        source_records.extend((split, record) for record in records)

    lengths = [len(record["text"]) for _, record in source_records]
    length_thresholds = quantile_thresholds(lengths, (0.20, 0.40, 0.60, 0.80))
    exact_counts = Counter(record["text"] for _, record in source_records)
    templates = {record["hash"]: normalise_template(record["text"]) for _, record in source_records}
    template_prefixes = {
        record_hash: template[:60] if len(template) >= 60 else template
        for record_hash, template in templates.items()
    }
    prefix_counts = Counter(template_prefixes.values())

    population = []
    for split, record in source_records:
        template_prefix = template_prefixes[record["hash"]]
        if exact_counts[record["text"]] > 1:
            repeat_kind = "exact_repeat"
        elif len(template_prefix) >= 60 and prefix_counts[template_prefix] > 1:
            repeat_kind = "near_template"
        else:
            repeat_kind = "unique"
        population.append(
            {
                "hash": record["hash"],
                "text": record["text"],
                "split": split,
                "entity_signature": entity_signature(record),
                "script": detect_script(record["text"]),
                "length_quantile": quantile_bin(len(record["text"]), length_thresholds),
                "entity_count_bin": entity_count_bin(len(record["entities"])),
                "dated_hash": bool(re.search(r"20\d{6}$", record["hash"])),
                "repeat_kind": repeat_kind,
                "template_group_size": prefix_counts[template_prefix],
                "char_count": len(record["text"]),
            }
        )
    return population, {"sources": sources, "length_quantile_thresholds": length_thresholds}


def apportion_integer_capacity(weights: dict[str, int], target: int, seed: int) -> dict[str, int]:
    total = sum(weights.values())
    exact = {key: target * value / total for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in exact.items()}
    remaining = target - sum(result.values())
    order = sorted(
        weights,
        key=lambda key: (exact[key] - result[key], stable_fraction(key, seed)),
        reverse=True,
    )
    for key in order[:remaining]:
        result[key] += 1
    return result


def assign_main_workers(
    main_items: list[dict[str, Any]], calibration_items: list[dict[str, Any]]
) -> dict[str, list[str]]:
    """Assign all non-calibration documents while balancing total run workload."""
    workers = [f"{number:02d}" for number in range(1, 21)]
    duplicate_target = round(0.20 * len(main_items))
    calibration_assignment_count = 4 * len(calibration_items)
    total_assignments = calibration_assignment_count + len(main_items) + duplicate_target
    if total_assignments % len(workers):
        raise ValueError("total assignments must divide evenly across 20 workers")
    total_capacity = total_assignments // len(workers)
    main_capacity = {
        worker: total_capacity - (len(calibration_items) if int(worker) <= 4 else 0)
        for worker in workers
    }
    primary_capacity = apportion_integer_capacity(main_capacity, len(main_items), SEED + 90)
    secondary_capacity = {
        worker: main_capacity[worker] - primary_capacity[worker] for worker in workers
    }
    if sum(secondary_capacity.values()) != duplicate_target:
        raise AssertionError("secondary capacities do not match duplicate target")

    assignments: dict[str, list[str]] = {item["item_id"]: [] for item in main_items}
    worker_counts = Counter()
    worker_chars = Counter({
        worker: sum(item["char_count"] for item in calibration_items) if int(worker) <= 4 else 0
        for worker in workers
    })
    worker_strata: dict[str, Counter] = {worker: Counter() for worker in workers}
    ordered = sorted(
        main_items,
        key=lambda item: (
            -item["char_count"],
            stable_fraction(item["item_id"], SEED + 100),
        ),
    )
    for item in ordered:
        key = stratum_key(item)
        candidates = [
            worker for worker in workers if worker_counts[worker] < primary_capacity[worker]
        ]
        worker = min(
            candidates,
            key=lambda candidate: (
                worker_strata[candidate][key],
                worker_chars[candidate],
                worker_counts[candidate],
                stable_fraction(item["item_id"] + candidate, SEED + 101),
            ),
        )
        assignments[item["item_id"]].append(worker)
        worker_counts[worker] += 1
        worker_chars[worker] += item["char_count"]
        worker_strata[worker][key] += 1

    duplicate_items, _ = stratified_sample(
        main_items,
        target=duplicate_target,
        seed=SEED + 200,
        rare_quota_fraction=0.10,
    )
    secondary_counts = Counter()
    secondary_strata: dict[str, Counter] = {worker: Counter() for worker in workers}
    duplicate_items.sort(
        key=lambda item: (-item["char_count"], stable_fraction(item["item_id"], SEED + 201))
    )
    for item in duplicate_items:
        key = stratum_key(item)
        first_worker = assignments[item["item_id"]][0]
        candidates = [
            worker
            for worker in workers
            if worker != first_worker and secondary_counts[worker] < secondary_capacity[worker]
        ]
        worker = min(
            candidates,
            key=lambda candidate: (
                secondary_strata[candidate][key],
                worker_chars[candidate],
                secondary_counts[candidate],
                stable_fraction(item["item_id"] + candidate, SEED + 202),
            ),
        )
        assignments[item["item_id"]].append(worker)
        secondary_counts[worker] += 1
        worker_chars[worker] += item["char_count"]
        secondary_strata[worker][key] += 1

    totals = Counter(worker for assigned in assignments.values() for worker in assigned)
    for worker in workers:
        expected_main = main_capacity[worker]
        if totals[worker] != expected_main:
            raise AssertionError(
                f"unbalanced main assignments for {worker}: {totals[worker]} != {expected_main}"
            )
    return assignments


def taxonomy_payload() -> dict[str, Any]:
    return {
        "version": TAXONOMY_VERSION,
        "created_for": "ner_uz_hackathon_participant style/source analysis",
        "run_model": RUN_MODEL,
        "run_reasoning_effort": RUN_REASONING_EFFORT,
        "calibration": "skipped_by_user_request",
        "annotation_unit": "one source document",
        "rules": {
            "genre": "Exactly one primary communicative genre.",
            "topics": "One to three topics; primary topic first.",
            "text_form": "Exactly one observed production form. Native_text is the conservative default when no OCR/transcript evidence is visible.",
            "register": "Exactly one register label.",
            "evidence": "Zero to three short exact substrings from the input text.",
            "confidence": "Numbers in [0, 1].",
            "mixed_composite": "Use only for an actual concatenation of different materials/authors, not merely bilingual text.",
            "genre_decision_order": [
                "First identify explicit specialised forms: business_response, customer_review, classified_listing, advertisement, quiz_poll, reference_result, personal_message.",
                "Then distinguish first-party official_announcement from third-person news_report.",
                "Use comment_reply for a context-dependent user reaction or conversational fragment.",
                "Use social_post only as the remaining standalone social publication/caption.",
                "Use other only when communicative function is genuinely indeterminate.",
            ],
            "anti_catchall_rules": [
                "Publication on Telegram or another social platform is a channel clue, not a reason by itself to choose social_post.",
                "A headline/date/byline followed by factual third-person reporting is news_report even if social links appear at the end.",
                "A price, route, contact, product parameters, or a concrete private offer normally indicates classified_listing; broad brand promotion indicates advertisement.",
                "A short score, schedule, track title, username list, profile, or other non-narrative metadata is reference_result, not comment_reply or social_post.",
                "A thank-you or evaluation of a doctor, restaurant, product, course, or service is customer_review; an organisation replying to it is business_response.",
                "A greeting or congratulations addressed to a person is personal_message.",
                "Choose topics from the semantic subject of the text, never from incidental brand/location/entity names. Use other only when no listed subject applies.",
                "If an explicit marker says 'Текст на изображении:' or equivalent, choose image_ocr. Repeated 'Спикер N:' or equivalent indicates speech_transcript.",
            ],
            "register_boundaries": {
                "formal": "Institutional/professional wording and consistently formal syntax.",
                "neutral": "Standard factual prose without strong institutional or colloquial marking.",
                "informal": "Conversational address, slang, expressive spelling, heavy emoji, or clearly casual language.",
                "mixed": "Two substantial sections with contrasting registers; isolated informal words do not suffice.",
            },
        },
        "genre": {"allowed": GENRES, "definitions": GENRE_DEFINITIONS},
        "topics": {"allowed": TOPICS, "definitions": TOPIC_DEFINITIONS},
        "text_form": {"allowed": TEXT_FORMS, "definitions": TEXT_FORM_DEFINITIONS},
        "register": {"allowed": REGISTERS, "definitions": REGISTER_DEFINITIONS},
        "output_schema": {
            "item_id": "string",
            "genre": "one genre label",
            "topics": "array of 1-3 unique topic labels",
            "text_form": "one text_form label",
            "register": "one register label",
            "evidence": "array of 0-3 exact input substrings",
            "genre_confidence": "number 0..1",
            "topics_confidence": "number 0..1",
        },
        "known_schema_limitation": {
            "channel": "The requested output schema defines no channel field or channel taxonomy. Agreement is therefore computed for text_form in its place; channel is reported as unavailable rather than invented post hoc."
        },
        "prior_run_audit_refinement": {
            "reason": "The archived Luna run showed catch-all use of social_post/comment_reply/other and confusion among topic and register boundaries.",
            "change": "Added explicit anti-catchall rules while preserving the requested label schema. No calibration stage is run in this Sol experiment.",
        },
    }


def refresh_taxonomy() -> None:
    write_json(ARTIFACT_DIR / "taxonomy.json", taxonomy_payload())
    print(ARTIFACT_DIR / "taxonomy.json")


def prepare() -> None:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    existing_annotations = list(ANNOTATION_DIR.glob("worker_*.jsonl"))
    if existing_annotations:
        raise RuntimeError(
            "refusing to overwrite an existing annotation run; move/remove artifacts/style_analysis/annotations first"
        )
    # Remove only stale, script-generated batch files from an unstarted run.
    for path in BATCH_DIR.glob("worker_*.jsonl"):
        path.unlink()

    population, source_metadata = build_population()
    calibration: list[dict[str, Any]] = []
    calibration_hashes: set[str] = set()
    main = population
    selected = population
    ordered_for_ids = sorted(
        selected, key=lambda item: stable_fraction(item["hash"], SEED + 20)
    )
    for index, item in enumerate(ordered_for_ids, start=1):
        item["item_id"] = f"sa_{index:06d}"

    assignments = assign_main_workers(main, calibration)

    item_by_id = {item["item_id"]: item for item in selected}
    for worker_number in range(1, 21):
        worker = f"{worker_number:02d}"
        worker_ids = sorted(
            [
                item_id
                for item_id, assigned in assignments.items()
                if worker in assigned and item_by_id[item_id]["hash"] not in calibration_hashes
            ],
            key=lambda item_id: stable_fraction(item_id, SEED + 40 + worker_number),
        )
        for part_index, start in enumerate(range(0, len(worker_ids), BATCH_SIZE), start=1):
            part_ids = worker_ids[start : start + BATCH_SIZE]
            write_jsonl(
                BATCH_DIR / f"worker_{worker}_main_part{part_index:02d}.jsonl",
                ({"item_id": item_id, "text": item_by_id[item_id]["text"]} for item_id in part_ids),
            )

    assignment_summaries = {}
    for worker_number in range(1, 21):
        worker = f"{worker_number:02d}"
        ids = [item_id for item_id, assigned in assignments.items() if worker in assigned]
        assignment_summaries[worker] = {
            "documents": len(ids),
            "total_characters": sum(item_by_id[item_id]["char_count"] for item_id in ids),
            "calibration_documents": 0,
            "main_documents": len(ids),
            "batch_files": sorted(path.name for path in BATCH_DIR.glob(f"worker_{worker}*.jsonl")),
        }

    manifest_items = []
    for item in sorted(selected, key=lambda value: value["item_id"]):
        key = stratum_key(item)
        stage = "main"
        manifest_items.append(
            {
                "item_id": item["item_id"],
                "hash": item["hash"],
                "split": item["split"],
                "sampling_stage": stage,
                "stratum": serialise_stratum(key),
                "template_group_size": item["template_group_size"],
                "char_count": item["char_count"],
                "calibration_selection_probability": 0.0,
                "prediction_inclusion_probability": 1.0,
                "design_weight": 1.0,
                "assigned_workers": assignments[item["item_id"]],
            }
        )

    manifest = {
        "version": "2.0.0",
        "seed": SEED,
        "run_model": RUN_MODEL,
        "run_reasoning_effort": RUN_REASONING_EFFORT,
        **source_metadata,
        "population_documents": len(population),
        "sampling_design": {
            "stratum_fields": [
                "split",
                "entity_signature",
                "script",
                "length_quantile",
                "entity_count_bin",
                "dated_hash",
                "repeat_kind",
            ],
            "calibration_documents": 0,
            "main_unique_documents": len(main),
            "main_double_annotated_documents": round(0.20 * len(main)),
            "calibration": "Skipped by explicit user request.",
            "batch_size_limit": BATCH_SIZE,
            "main_coverage": "All train/dev records are in the main stage; there is no calibration subset.",
            "near_template_definition": "Same first 60 characters after casefolding and replacement of URLs, handles, phone-like strings, numbers, punctuation, and whitespace.",
            "estimation": "All corpus records receive a prediction, so prediction inclusion probability and design weight are both 1.",
        },
        "agent_visibility": "Batch files expose item_id and text only. Gold entities and source hash are absent.",
        "assignment_summaries": assignment_summaries,
        "items": manifest_items,
    }
    write_json(ARTIFACT_DIR / "taxonomy.json", taxonomy_payload())
    write_json(ARTIFACT_DIR / "sampling_manifest.json", manifest)
    print(json.dumps({
        "population": len(population),
        "calibration": len(calibration),
        "main": len(main),
        "total_assignment_count": sum(v["documents"] for v in assignment_summaries.values()),
        "artifact_dir": str(ARTIFACT_DIR),
    }, ensure_ascii=False, indent=2))


def expected_worker_items(worker: str, stage: str = "all") -> dict[str, str]:
    if stage == "all":
        pattern = f"worker_{worker}_*.jsonl"
    elif stage in {"calibration", "main"}:
        pattern = f"worker_{worker}_{stage}_*.jsonl"
    else:
        raise ValueError(f"unknown validation stage: {stage}")
    paths = sorted(BATCH_DIR.glob(pattern))
    expected = {}
    for path in paths:
        for row in read_jsonl(path):
            item_id = row.get("item_id")
            text = row.get("text")
            if not isinstance(item_id, str) or not isinstance(text, str):
                raise ValueError(f"invalid batch row in {path}")
            if item_id in expected:
                raise ValueError(f"duplicate expected item_id {item_id} for worker {worker}")
            expected[item_id] = text
    return expected


def validate_worker(worker: str, stage: str = "all") -> dict[str, Any]:
    expected = expected_worker_items(worker, stage)
    path = ANNOTATION_DIR / f"worker_{worker}.jsonl"
    errors = []
    rows = []
    if not path.exists():
        return {
            "worker": worker,
            "stage": stage,
            "valid": False,
            "expected": len(expected),
            "received": 0,
            "errors": [f"missing file: {path}"],
        }
    try:
        rows = read_jsonl(path)
    except ValueError as error:
        return {
            "worker": worker,
            "stage": stage,
            "valid": False,
            "expected": len(expected),
            "received": 0,
            "errors": [str(error)],
        }

    counts = Counter(row.get("item_id") for row in rows)
    for item_id, count in counts.items():
        if count != 1:
            errors.append(f"item_id {item_id!r} occurs {count} times")
    missing = sorted(set(expected) - set(counts))
    extra = sorted(set(counts) - set(expected))
    if missing:
        errors.append(f"missing item_ids ({len(missing)}): {missing[:20]}")
    if extra:
        errors.append(f"extra item_ids ({len(extra)}): {extra[:20]}")

    for line_number, row in enumerate(rows, start=1):
        prefix = f"line {line_number}"
        item_id = row.get("item_id")
        if set(row) != OUTPUT_FIELDS:
            errors.append(
                f"{prefix}: fields differ: missing={sorted(OUTPUT_FIELDS-set(row))}, extra={sorted(set(row)-OUTPUT_FIELDS)}"
            )
        if row.get("genre") not in GENRES:
            errors.append(f"{prefix}: invalid genre {row.get('genre')!r}")
        topics = row.get("topics")
        if not isinstance(topics, list) or not 1 <= len(topics) <= 3:
            errors.append(f"{prefix}: topics must contain 1-3 labels")
        elif len(set(topics)) != len(topics) or any(topic not in TOPICS for topic in topics):
            errors.append(f"{prefix}: invalid or duplicate topics {topics!r}")
        if row.get("text_form") not in TEXT_FORMS:
            errors.append(f"{prefix}: invalid text_form {row.get('text_form')!r}")
        if row.get("register") not in REGISTERS:
            errors.append(f"{prefix}: invalid register {row.get('register')!r}")
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or len(evidence) > 3 or any(not isinstance(part, str) for part in evidence):
            errors.append(f"{prefix}: evidence must be an array of at most 3 strings")
        elif item_id in expected:
            for part in evidence:
                if part not in expected[item_id]:
                    errors.append(f"{prefix}: evidence is not an exact substring: {part!r}")
        for field in ("genre_confidence", "topics_confidence"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                errors.append(f"{prefix}: {field} must be a number in [0, 1]")
    return {
        "worker": worker,
        "stage": stage,
        "valid": not errors,
        "expected": len(expected),
        "received": len(rows),
        "errors": errors[:200],
        "error_count": len(errors),
    }


def validate_all(stage: str = "all") -> dict[str, Any]:
    worker_numbers = range(1, 5) if stage == "calibration" else range(1, 21)
    reports = [validate_worker(f"{number:02d}", stage) for number in worker_numbers]
    payload = {
        "stage": stage,
        "all_valid": all(report["valid"] for report in reports),
        "workers": reports,
    }
    write_json(ARTIFACT_DIR / "validation_report.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def categorical_majority(values: list[str]) -> str | None:
    counts = Counter(values)
    value, count = counts.most_common(1)[0]
    return value if count > len(values) / 2 else None


def topic_majority(values: list[list[str]]) -> list[str] | None:
    counts = Counter(topic for topics in values for topic in set(topics))
    winners = [topic for topic, count in counts.items() if count > len(values) / 2]
    if not winners:
        return None
    primary_counts = Counter(topics[0] for topics in values)
    winners.sort(key=lambda topic: (-primary_counts[topic], -counts[topic], TOPICS.index(topic)))
    return winners[:3]


def cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(left_counts[label] * right_counts[label] for label in set(left) | set(right)) / len(left) ** 2
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def nominal_alpha(ratings_by_item: list[list[str]]) -> float | None:
    ratings_by_item = [ratings for ratings in ratings_by_item if len(ratings) >= 2]
    if not ratings_by_item:
        return None
    disagreeing_pairs = 0
    total_pairs = 0
    all_ratings = []
    for ratings in ratings_by_item:
        for left_index in range(len(ratings)):
            for right_index in range(left_index + 1, len(ratings)):
                total_pairs += 1
                disagreeing_pairs += ratings[left_index] != ratings[right_index]
        all_ratings.extend(ratings)
    observed_disagreement = disagreeing_pairs / total_pairs
    counts = Counter(all_ratings)
    total = len(all_ratings)
    expected_disagreement = 1 - sum(count * (count - 1) for count in counts.values()) / (total * (total - 1))
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return 1 - observed_disagreement / expected_disagreement


def confusion_matrix(left: list[str], right: list[str], labels: list[str]) -> dict[str, dict[str, int]]:
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for a, b in zip(left, right):
        matrix[a][b] += 1
    return matrix


def agreement_for_group(
    item_annotations: dict[str, list[dict[str, Any]]],
    worker_ids: list[str],
) -> dict[str, Any]:
    dimensions = {
        "genre": GENRES,
        "text_form": TEXT_FORMS,
        "register": REGISTERS,
    }
    result: dict[str, Any] = {}
    for field, labels in dimensions.items():
        ratings_by_item = [
            [annotation[field] for annotation in annotations]
            for annotations in item_annotations.values()
            if len(annotations) >= 2
        ]
        pairwise = {}
        for left_index, left_worker in enumerate(worker_ids):
            for right_worker in worker_ids[left_index + 1 :]:
                left_values = []
                right_values = []
                for annotations in item_annotations.values():
                    by_worker = {annotation["_worker"]: annotation for annotation in annotations}
                    if left_worker in by_worker and right_worker in by_worker:
                        left_values.append(by_worker[left_worker][field])
                        right_values.append(by_worker[right_worker][field])
                if left_values:
                    pairwise[f"{left_worker}_vs_{right_worker}"] = {
                        "documents": len(left_values),
                        "cohen_kappa": cohen_kappa(left_values, right_values),
                        "confusion_matrix": confusion_matrix(left_values, right_values, labels),
                    }
        exact_matches = sum(len(set(ratings)) == 1 for ratings in ratings_by_item)
        result[field] = {
            "documents_with_multiple_ratings": len(ratings_by_item),
            "exact_agreement": exact_matches / len(ratings_by_item) if ratings_by_item else None,
            "krippendorff_alpha_nominal": nominal_alpha(ratings_by_item),
            "pairwise": pairwise,
        }

    topic_pairs = []
    for annotations in item_annotations.values():
        for left_index in range(len(annotations)):
            for right_index in range(left_index + 1, len(annotations)):
                left = set(annotations[left_index]["topics"])
                right = set(annotations[right_index]["topics"])
                topic_pairs.append({
                    "exact": left == right,
                    "jaccard": len(left & right) / len(left | right),
                })
    result["topics"] = {
        "rating_pairs": len(topic_pairs),
        "exact_agreement": statistics.mean(pair["exact"] for pair in topic_pairs) if topic_pairs else None,
        "mean_jaccard": statistics.mean(pair["jaccard"] for pair in topic_pairs) if topic_pairs else None,
    }
    return result


def calibration_report() -> None:
    payload = {
        "taxonomy_version": TAXONOMY_VERSION,
        "status": "calibration_skipped_by_user_request",
        "calibration": None,
        "channel": {
            "available": False,
            "reason": "The required output schema contains no channel field or allowed channel values.",
        },
        "documents_with_any_disagreement": 0,
    }
    write_json(ARTIFACT_DIR / "agreement_report.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def weighted_distribution(
    predictions: list[dict[str, Any]], field: str, labels: list[str]
) -> dict[str, Any]:
    weighted = Counter()
    unresolved_weight = 0.0
    for prediction in predictions:
        weight = prediction["design_weight"]
        value = prediction[field]
        if value is None:
            unresolved_weight += weight
        elif isinstance(value, list):
            for label in value:
                weighted[label] += weight
        else:
            weighted[value] += weight
    denominator = sum(weighted.values())
    return {
        "weighted_counts": {label: weighted[label] for label in labels},
        "share_among_resolved": {
            label: (weighted[label] / denominator if denominator else None) for label in labels
        },
        "unresolved_design_weight": unresolved_weight,
    }


def merge_and_report() -> None:
    validation = validate_all("all")
    if not validation["all_valid"]:
        raise RuntimeError("worker annotations are not valid; inspect validation_report.json")

    manifest = json.loads((ARTIFACT_DIR / "sampling_manifest.json").read_text(encoding="utf-8"))
    manifest_by_id = {item["item_id"]: item for item in manifest["items"]}
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for worker_number in range(1, 21):
        worker = f"{worker_number:02d}"
        for annotation in read_jsonl(ANNOTATION_DIR / f"worker_{worker}.jsonl"):
            annotation = dict(annotation)
            annotation["_worker"] = worker
            by_item[annotation["item_id"]].append(annotation)

    merged_rows = []
    predictions = []
    for item_id in sorted(manifest_by_id):
        annotations = sorted(by_item[item_id], key=lambda row: row["_worker"])
        n = len(annotations)
        genre = annotations[0]["genre"] if n == 1 else categorical_majority([a["genre"] for a in annotations])
        topics = annotations[0]["topics"] if n == 1 else topic_majority([a["topics"] for a in annotations])
        text_form = annotations[0]["text_form"] if n == 1 else categorical_majority([a["text_form"] for a in annotations])
        register = annotations[0]["register"] if n == 1 else categorical_majority([a["register"] for a in annotations])
        conflict_fields = []
        for field in ("genre", "text_form", "register"):
            if len({annotation[field] for annotation in annotations}) > 1:
                conflict_fields.append(field)
        if len({tuple(annotation["topics"]) for annotation in annotations}) > 1:
            conflict_fields.append("topics")
        unresolved_fields = [
            field
            for field, value in (
                ("genre", genre),
                ("topics", topics),
                ("text_form", text_form),
                ("register", register),
            )
            if value is None
        ]
        evidence = []
        for annotation in annotations:
            for excerpt in annotation["evidence"]:
                if excerpt not in evidence:
                    evidence.append(excerpt)
                if len(evidence) == 3:
                    break
            if len(evidence) == 3:
                break
        metadata = manifest_by_id[item_id]
        clean_annotations = [
            {key: value for key, value in annotation.items() if key != "_worker"} | {"worker": annotation["_worker"]}
            for annotation in annotations
        ]
        resolved = {
            "item_id": item_id,
            "genre": genre,
            "topics": topics,
            "text_form": text_form,
            "register": register,
            "evidence": evidence,
            "genre_confidence": statistics.mean(a["genre_confidence"] for a in annotations),
            "topics_confidence": statistics.mean(a["topics_confidence"] for a in annotations),
            "has_conflict": bool(conflict_fields),
            "conflict_fields": conflict_fields,
            "needs_review": bool(unresolved_fields),
            "unresolved_fields": unresolved_fields,
        }
        merged_rows.append({
            "item_id": item_id,
            "hash": metadata["hash"],
            "split": metadata["split"],
            "sampling_stage": metadata["sampling_stage"],
            "annotations": clean_annotations,
            "resolved": resolved,
        })
        predictions.append({
            **resolved,
            "hash": metadata["hash"],
            "split": metadata["split"],
            "sampling_stage": metadata["sampling_stage"],
            "prediction_inclusion_probability": metadata["prediction_inclusion_probability"],
            "design_weight": metadata["design_weight"],
        })

    calibration_annotations = {
        item_id: annotations
        for item_id, annotations in by_item.items()
        if manifest_by_id[item_id]["sampling_stage"] == "calibration"
    }
    main_duplicate_annotations = {
        item_id: annotations
        for item_id, annotations in by_item.items()
        if manifest_by_id[item_id]["sampling_stage"] == "main" and len(annotations) == 2
    }
    agreement = {
        "taxonomy_version": TAXONOMY_VERSION,
        "run_model": RUN_MODEL,
        "run_reasoning_effort": RUN_REASONING_EFFORT,
        "calibration": {
            "status": "skipped_by_user_request",
            "documents": 0,
        },
        "main_double_annotation": agreement_for_group(main_duplicate_annotations, [f"{n:02d}" for n in range(1, 21)]),
        "channel": {
            "available": False,
            "reason": "No channel field or allowed channel values were supplied in the required annotation schema. Text-form agreement is reported instead.",
        },
    }
    quality = {
        "scope": {
            "population_documents": manifest["population_documents"],
            "llm_annotated_unique_documents": len(predictions),
            "full_predictions_scope": "All 14,500 train/dev documents were annotated by gpt-5.6-sol (low); no heuristic corpus-wide labels were generated.",
            "calibration": "skipped_by_user_request",
            "batch_size_limit": BATCH_SIZE,
        },
        "validation": validation,
        "conflicts": {
            "documents_with_any_conflict": sum(row["has_conflict"] for row in predictions),
            "documents_needing_review": sum(row["needs_review"] for row in predictions),
            "by_field": dict(Counter(field for row in predictions for field in row["conflict_fields"])),
        },
        "weighted_estimates": {
            "genre": weighted_distribution(predictions, "genre", GENRES),
            "topics": weighted_distribution(predictions, "topics", TOPICS),
            "text_form": weighted_distribution(predictions, "text_form", TEXT_FORMS),
            "register": weighted_distribution(predictions, "register", REGISTERS),
        },
        "manual_review": {
            "status": "pending",
            "required_random_examples": 100,
            "seed": SEED,
        },
    }
    write_jsonl(ARTIFACT_DIR / "merged_annotations.jsonl", merged_rows)
    write_jsonl(ARTIFACT_DIR / "full_predictions.jsonl", predictions)
    write_json(ARTIFACT_DIR / "agreement_report.json", agreement)
    write_json(ARTIFACT_DIR / "quality_report.json", quality)
    print(json.dumps({
        "merged_documents": len(merged_rows),
        "conflicts": quality["conflicts"],
        "agreement_report": str(ARTIFACT_DIR / "agreement_report.json"),
    }, ensure_ascii=False, indent=2))


def export_qa() -> None:
    predictions = read_jsonl(ARTIFACT_DIR / "full_predictions.jsonl")
    manifest = json.loads((ARTIFACT_DIR / "sampling_manifest.json").read_text(encoding="utf-8"))
    manifest_by_id = {item["item_id"]: item for item in manifest["items"]}
    text_by_id = {}
    for worker_number in range(1, 21):
        worker = f"{worker_number:02d}"
        for item_id, text in expected_worker_items(worker).items():
            text_by_id[item_id] = text
    random_sample = sorted(
        predictions,
        key=lambda row: stable_fraction(row["item_id"], SEED + 500),
    )[:100]
    conflict_examples = {}
    for field in ("genre", "topics", "text_form", "register"):
        candidates = [row for row in predictions if field in row["conflict_fields"]]
        conflict_examples[field] = [row["item_id"] for row in candidates[:20]]
    qa_rows = []
    for row in random_sample:
        qa_rows.append({
            "item_id": row["item_id"],
            "hash": manifest_by_id[row["item_id"]]["hash"],
            "text": text_by_id[row["item_id"]],
            "prediction": {key: row[key] for key in ("genre", "topics", "text_form", "register", "evidence", "needs_review")},
            "manual_verdict": None,
            "manual_notes": "",
        })
    write_jsonl(ARTIFACT_DIR / "qa_review_sample.jsonl", qa_rows)
    write_json(ARTIFACT_DIR / "qa_conflict_index.json", conflict_examples)
    print(json.dumps({
        "random_examples": len(qa_rows),
        "conflict_example_counts": {key: len(value) for key, value in conflict_examples.items()},
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare", "refresh-taxonomy", "validate", "calibration-report", "merge", "export-qa")
    )
    parser.add_argument("--worker", help="two-digit worker number for validation")
    parser.add_argument(
        "--stage", choices=("all", "calibration", "main"), default="all",
        help="batch subset used by validation",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "refresh-taxonomy":
        refresh_taxonomy()
    elif args.command == "validate":
        if args.worker:
            print(json.dumps(validate_worker(args.worker, args.stage), ensure_ascii=False, indent=2))
        else:
            validate_all(args.stage)
    elif args.command == "calibration-report":
        calibration_report()
    elif args.command == "merge":
        merge_and_report()
    elif args.command == "export-qa":
        export_qa()


if __name__ == "__main__":
    main()
