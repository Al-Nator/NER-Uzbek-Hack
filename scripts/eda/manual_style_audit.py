#!/usr/bin/env python3
"""Persist the primary-agent audit of the reproducible Sol style annotations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "style_analysis"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


# The primary agent inspected all 100 seeded examples. Only material mismatches
# are listed here; every unlisted seeded example was accepted.
RANDOM_ISSUES = {
    "sa_000716": {
        "fields": ["genre", "register"],
        "suggested": {"genre": "mixed_composite", "register": "mixed"},
        "note": "A profile fragment is concatenated with a separate conversational reflection.",
    },
    "sa_001023": {
        "fields": ["topics"],
        "suggested": {"topics": ["society_social_issues"]},
        "note": "The benefit assignment is substantive; bot/interface text is incidental OCR context.",
    },
    "sa_001225": {
        "fields": ["topics"],
        "suggested": {"topics": ["other"]},
        "note": "Minimal metadata does not establish a listed subject; topics cannot be null.",
    },
    "sa_002582": {
        "fields": ["topics"],
        "suggested": {"topics": ["international_relations_conflict"]},
        "note": "The assertions concern interstate military conflict, not a distinct crime topic.",
    },
    "sa_003176": {
        "fields": ["register"],
        "suggested": {"register": "neutral"},
        "note": "The brief welcome is unmarked; register cannot be null.",
    },
    "sa_004111": {
        "fields": ["genre"],
        "suggested": {"genre": "news_report"},
        "note": "Detached third-person reporting lacks a clear first-party institutional voice.",
    },
    "sa_004984": {
        "fields": ["topics"],
        "suggested": {"topics": ["health_medicine"]},
        "note": "The cosmetology metadata supports medicine, not entertainment/fandom.",
    },
    "sa_005252": {
        "fields": ["genre"],
        "suggested": {"genre": "news_report"},
        "note": "This is narrative sports reporting, not a first-party official announcement.",
    },
    "sa_006002": {
        "fields": ["topics"],
        "suggested": {"topics": ["technology_telecom"]},
        "note": "Here 'report' denotes a platform report rather than law/crime/security.",
    },
    "sa_007050": {
        "fields": ["topics"],
        "suggested": {"topics": ["international_relations_conflict"]},
        "note": "Military aircraft are instruments of the conflict, not a separate transport topic.",
    },
    "sa_007107": {
        "fields": ["genre"],
        "suggested": {"genre": "comment_reply"},
        "note": "This is a contextual tariff question, not an evaluative customer review.",
    },
    "sa_007555": {
        "fields": ["genre"],
        "suggested": {"genre": "advertisement"},
        "note": "The text promotes a concessional loan and provides product terms.",
    },
    "sa_009453": {
        "fields": ["topics"],
        "suggested": {"topics": ["environment_agriculture"]},
        "note": "The comment concerns livestock; consumer services is unsupported.",
    },
    "sa_011370": {
        "fields": ["register"],
        "suggested": {"register": "neutral"},
        "note": "The media metadata is unmarked; register cannot be null.",
    },
    "sa_011910": {
        "fields": ["topics"],
        "suggested": {"topics": ["environment_agriculture", "politics_government"]},
        "note": "The construction company is an incidental entity, not a substantive topic.",
    },
    "sa_012388": {
        "fields": ["topics"],
        "suggested": {"topics": ["employment_migration", "business_economy_finance"]},
        "note": "The report concerns wage arrears and repayment; the stadium is only the workplace.",
    },
    "sa_013360": {
        "fields": ["genre"],
        "suggested": {"genre": "comment_reply"},
        "note": "This is a readable context-dependent social fragment, not an indeterminate genre.",
    },
}


# Deterministic adjudications for the seeded conflict audit. These are an audit
# layer: raw worker annotations and majority-vote outputs remain untouched.
CONFLICT_RESOLUTIONS = {
    "genre": {
        "sa_000124": "other",
        "sa_000133": "business_response",
        "sa_000168": "customer_review",
        "sa_000192": "other",
        "sa_000211": "personal_message",
        "sa_000238": "other",
        "sa_000253": "comment_reply",
        "sa_000268": "personal_message",
        "sa_000313": "comment_reply",
        "sa_000336": "reference_result",
        "sa_000368": "advertisement",
        "sa_000469": "comment_reply",
        "sa_000480": "classified_listing",
        "sa_000518": "social_post",
        "sa_000540": "customer_review",
        "sa_000583": "classified_listing",
        "sa_000590": "comment_reply",
        "sa_000602": "personal_message",
        "sa_000632": "reference_result",
        "sa_000716": "mixed_composite",
    },
    "topics": {
        "sa_000051": ["education_science", "religion_spirituality"],
        "sa_000053": ["business_economy_finance", "education_science", "personal_daily_life"],
        "sa_000069": ["politics_government", "law_crime_security", "international_relations_conflict"],
        "sa_000093": ["religion_spirituality", "international_relations_conflict", "transport_automotive_aviation"],
        "sa_000094": ["other"],
        "sa_000096": ["construction_real_estate_urbanism", "religion_spirituality", "personal_daily_life"],
        "sa_000098": ["employment_migration", "law_crime_security"],
        "sa_000101": ["education_science", "technology_telecom"],
        "sa_000120": ["society_social_issues", "religion_spirituality", "culture_history_language"],
        "sa_000133": ["other"],
        "sa_000160": ["other"],
        "sa_000163": ["politics_government"],
        "sa_000181": ["sports_esports", "business_economy_finance"],
        "sa_000211": ["society_social_issues", "employment_migration", "personal_daily_life"],
        "sa_000219": ["consumer_goods_services"],
        "sa_000221": ["health_medicine", "sports_esports"],
        "sa_000230": ["education_science", "employment_migration"],
        "sa_000253": ["law_crime_security", "politics_government"],
        "sa_000290": ["entertainment_music_fandom", "consumer_goods_services", "culture_history_language"],
        "sa_000309": ["society_social_issues", "health_medicine"],
    },
    "text_form": {
        "sa_001338": "native_text",
        "sa_001575": "native_text",
        "sa_002635": "speech_transcript",
        "sa_003329": "native_text",
        "sa_004284": "speech_transcript",
        "sa_005851": "speech_transcript",
        "sa_005961": "native_text",
        "sa_006255": "speech_transcript",
        "sa_006694": "native_text",
        "sa_013091": "speech_transcript",
    },
    "register": {
        "sa_000087": "neutral",
        "sa_000101": "informal",
        "sa_000204": "formal",
        "sa_000221": "neutral",
        "sa_000223": "neutral",
        "sa_000230": "neutral",
        "sa_000238": "informal",
        "sa_000268": "informal",
        "sa_000278": "neutral",
        "sa_000290": "informal",
        "sa_000309": "formal",
        "sa_000385": "formal",
        "sa_000438": "neutral",
        "sa_000503": "neutral",
        "sa_000525": "informal",
        "sa_000567": "formal",
        "sa_000572": "informal",
        "sa_000590": "neutral",
        "sa_000632": "neutral",
        "sa_000716": "informal",
    },
}


def main() -> None:
    qa_path = ARTIFACTS / "qa_review_sample.jsonl"
    sample = read_jsonl(qa_path)
    if len(sample) != 100 or len({row["item_id"] for row in sample}) != 100:
        raise RuntimeError("expected exactly 100 unique seeded QA rows")
    if not set(RANDOM_ISSUES).issubset({row["item_id"] for row in sample}):
        raise RuntimeError("random-review decisions do not match the seeded sample")

    for row in sample:
        finding = RANDOM_ISSUES.get(row["item_id"])
        row["manual_verdict"] = "issue" if finding else "accept"
        row["manual_notes"] = (
            finding["note"] if finding else "Primary-agent review found no material mismatch."
        )
        row["manual_issue_fields"] = finding["fields"] if finding else []
        row["manual_suggested"] = finding["suggested"] if finding else {}
    write_jsonl(qa_path, sample)

    conflict_index = json.loads((ARTIFACTS / "qa_conflict_index.json").read_text(encoding="utf-8"))
    for field, decisions in CONFLICT_RESOLUTIONS.items():
        expected_ids = set(conflict_index[field])
        if set(decisions) != expected_ids:
            raise RuntimeError(f"{field} adjudications do not match qa_conflict_index.json")

    merged = {row["item_id"]: row for row in read_jsonl(ARTIFACTS / "merged_annotations.jsonl")}
    texts: dict[str, str] = {}
    for path in sorted((ARTIFACTS / "batches").glob("worker_*.jsonl")):
        for row in read_jsonl(path):
            texts[row["item_id"]] = row["text"]

    conflict_rows = []
    for field, decisions in CONFLICT_RESOLUTIONS.items():
        for item_id, resolution in decisions.items():
            source = merged[item_id]
            conflict_rows.append({
                "item_id": item_id,
                "field": field,
                "text": texts[item_id],
                "worker_values": [
                    {"worker": annotation["worker"], "value": annotation[field]}
                    for annotation in source["annotations"]
                ],
                "majority_vote_value": source["resolved"][field],
                "primary_agent_resolution": resolution,
                "review_status": "reviewed",
                "review_note": "Resolved from the source text using taxonomy v2.0.0; raw annotations are preserved.",
            })
    write_jsonl(ARTIFACTS / "qa_conflict_review.jsonl", conflict_rows)

    quality_path = ARTIFACTS / "quality_report.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    verdict_counts = Counter(row["manual_verdict"] for row in sample)
    field_counts = Counter(field for row in sample for field in row["manual_issue_fields"])
    quality["scope"]["full_predictions_scope"] = (
        "All 14,500 train/dev documents were LLM-annotated; no heuristic corpus-wide labels were generated."
    )
    issue_fields = {
        field: field_counts[field]
        for field in ("genre", "topics", "text_form", "register")
    }
    quality["manual_review"] = {
        "status": "completed",
        "seed": 42,
        "reviewer": "primary_agent",
        "random_examples_reviewed": 100,
        "random_sample_verdicts": dict(verdict_counts),
        "random_sample_issue_rate": verdict_counts["issue"] / 100,
        "random_sample_issue_fields": issue_fields,
        "conflict_examples_reviewed_by_field": {
            field: len(decisions) for field, decisions in CONFLICT_RESOLUTIONS.items()
        },
        "conflict_examples_reviewed_total": sum(
            len(decisions) for decisions in CONFLICT_RESOLUTIONS.values()
        ),
        "conflict_review_policy": (
            "Seeded examples from every conflict field: up to 20 per field; all 10 text_form conflicts."
        ),
        "calibration": "Skipped exactly as requested for this Sol run.",
        "interpretation": (
            "Double-annotation agreement is strong overall, but 17% of the seeded random audit had at least "
            "one material issue. Multi-label topic boundaries and unresolved 1-1 ties remain the main caveats; "
            "corpus shares should therefore be treated as exploratory estimates."
        ),
    }
    quality_path.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(quality["manual_review"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
