#!/usr/bin/env python3
"""
Dataset quality filter for NS-LLM style JSON files.

Checks two common issues in a single file:
1) Answers containing multiple emojis.
2) Duplicate questions within the same file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


QUESTION_FIELD_PRIORITY: List[List[str]] = [
    ["prompt"],
    ["instruction", "input"],
    ["instruction"],
    ["question"],
    ["query"],
    ["user"],
]

ANSWER_FIELD_PRIORITY: List[List[str]] = [
    ["completion"],
    ["output"],
    ["answer"],
    ["response"],
    ["assistant"],
]


# Match common emoji clusters (single emoji, modifiers, flags, ZWJ sequences).
EMOJI_PATTERN = re.compile(
    r"(?:"
    r"[\U0001F1E6-\U0001F1FF]{2}"
    r"|"
    r"(?:[\U0001F300-\U0001FAFF\u2600-\u27BF])"
    r"(?:\uFE0F|\uFE0E)?"
    r"(?:[\U0001F3FB-\U0001F3FF])?"
    r"(?:\u200D(?:[\U0001F300-\U0001FAFF\u2600-\u27BF])(?:\uFE0F|\uFE0E)?(?:[\U0001F3FB-\U0001F3FF])?)*"
    r")"
)


def parse_fields_arg(fields_arg: str) -> List[str]:
    fields = [field.strip() for field in fields_arg.split(",") if field.strip()]
    return fields


def load_json_items(file_path: Path) -> List[Any]:
    with file_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("data", "items", "dataset", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(
        "JSON root must be a list, or an object containing list field one of: data/items/dataset/records."
    )


def detect_field_combo(sample: Dict[str, Any], priorities: Sequence[Sequence[str]]) -> List[str]:
    for combo in priorities:
        present = [field for field in combo if field in sample]
        if combo == ["instruction", "input"] and "instruction" in sample:
            return ["instruction", "input"] if "input" in sample else ["instruction"]
        if present:
            return present

    fallback = [k for k, v in sample.items() if isinstance(v, (str, int, float, bool))]
    if fallback:
        return [fallback[0]]
    return []


def extract_text(item: Dict[str, Any], fields: Sequence[str]) -> str:
    parts: List[str] = []
    for field in fields:
        value = item.get(field, "")
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value:
            parts.append(value)
    return "\n".join(parts).strip()


def normalize_question(text: str, collapse_whitespace: bool = True) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip()
    if collapse_whitespace:
        normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def find_emojis(text: str) -> List[str]:
    if not text:
        return []
    return [match.group(0) for match in EMOJI_PATTERN.finditer(text)]


def shorten_text(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    if max_chars <= 1:
        return compact[:max_chars]
    return compact[: max_chars - 1] + "..."


def analyze_file(
    items: Sequence[Any],
    question_fields: Sequence[str],
    answer_fields: Sequence[str],
    preview_chars: int,
    collapse_ws_for_duplicate: bool,
) -> Dict[str, Any]:
    valid_rows = 0
    non_dict_rows = 0
    empty_question_rows = 0
    empty_answer_rows = 0

    zero_emoji_rows = 0
    single_emoji_rows = 0
    multi_emoji_rows = 0

    multi_emoji_examples: List[Dict[str, Any]] = []
    duplicate_map: Dict[str, Dict[str, Any]] = {}

    for idx_zero_based, raw_item in enumerate(items):
        idx = idx_zero_based + 1
        if not isinstance(raw_item, dict):
            non_dict_rows += 1
            continue

        valid_rows += 1
        question = extract_text(raw_item, question_fields)
        answer = extract_text(raw_item, answer_fields)

        if not question:
            empty_question_rows += 1
        else:
            normalized_question = normalize_question(
                question, collapse_whitespace=collapse_ws_for_duplicate
            )
            if normalized_question:
                entry = duplicate_map.setdefault(
                    normalized_question,
                    {
                        "question_preview": shorten_text(question, preview_chars),
                        "indexes": [],
                    },
                )
                entry["indexes"].append(idx)

        if not answer:
            empty_answer_rows += 1

        emojis = find_emojis(answer)
        emoji_count = len(emojis)

        if emoji_count == 0:
            zero_emoji_rows += 1
        elif emoji_count == 1:
            single_emoji_rows += 1
        else:
            multi_emoji_rows += 1
            multi_emoji_examples.append(
                {
                    "index": idx,
                    "emoji_count": emoji_count,
                    "emojis": emojis,
                    "question_preview": shorten_text(question, preview_chars),
                    "answer_preview": shorten_text(answer, preview_chars),
                }
            )

    duplicate_groups: List[Dict[str, Any]] = []
    for normalized_question, group in duplicate_map.items():
        indexes = group["indexes"]
        if len(indexes) > 1:
            duplicate_groups.append(
                {
                    "question_preview": group["question_preview"],
                    "normalized_question": normalized_question,
                    "occurrences": len(indexes),
                    "indexes": indexes,
                }
            )

    duplicate_groups.sort(key=lambda x: (-x["occurrences"], x["indexes"][0]))
    multi_emoji_examples.sort(key=lambda x: (-x["emoji_count"], x["index"]))

    duplicate_rows_excluding_first = sum(
        max(0, group["occurrences"] - 1) for group in duplicate_groups
    )

    return {
        "summary": {
            "total_rows": len(items),
            "valid_dict_rows": valid_rows,
            "non_dict_rows": non_dict_rows,
            "empty_question_rows": empty_question_rows,
            "empty_answer_rows": empty_answer_rows,
            "question_fields": list(question_fields),
            "answer_fields": list(answer_fields),
            "zero_emoji_rows": zero_emoji_rows,
            "single_emoji_rows": single_emoji_rows,
            "multi_emoji_rows": multi_emoji_rows,
            "duplicate_question_groups": len(duplicate_groups),
            "duplicate_question_rows_excluding_first": duplicate_rows_excluding_first,
        },
        "multi_emoji_examples": multi_emoji_examples,
        "duplicate_question_groups": duplicate_groups,
    }


def print_report(report: Dict[str, Any], max_examples: int) -> None:
    summary = report["summary"]
    print("[SUMMARY]")
    for key in (
        "total_rows",
        "valid_dict_rows",
        "non_dict_rows",
        "empty_question_rows",
        "empty_answer_rows",
        "question_fields",
        "answer_fields",
        "zero_emoji_rows",
        "single_emoji_rows",
        "multi_emoji_rows",
        "duplicate_question_groups",
        "duplicate_question_rows_excluding_first",
    ):
        print(f"- {key}: {summary[key]}")

    multi = report["multi_emoji_examples"]
    print("\n[MULTI_EMOJI_EXAMPLES]")
    if not multi:
        print("- No multi-emoji rows found.")
    else:
        print(f"- Showing {min(max_examples, len(multi))}/{len(multi)}")
        for item in multi[:max_examples]:
            emoji_compact = "".join(item["emojis"])
            print(
                f"- index={item['index']} emoji_count={item['emoji_count']} emojis={emoji_compact} "
                f"question={item['question_preview']}"
            )

    duplicate_groups = report["duplicate_question_groups"]
    print("\n[DUPLICATE_QUESTION_GROUPS]")
    if not duplicate_groups:
        print("- No duplicate question groups found.")
    else:
        print(f"- Showing {min(max_examples, len(duplicate_groups))}/{len(duplicate_groups)}")
        for group in duplicate_groups[:max_examples]:
            print(
                f"- occurrences={group['occurrences']} indexes={group['indexes']} "
                f"question={group['question_preview']}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Quickly filter dataset quality issues: multi-emoji answers and duplicate questions."
        )
    )
    parser.add_argument("input_file", type=str, help="Path to a JSON dataset file.")
    parser.add_argument(
        "--question-fields",
        type=str,
        default="",
        help="Comma-separated fields used as question text, e.g. 'prompt' or 'instruction,input'.",
    )
    parser.add_argument(
        "--answer-fields",
        type=str,
        default="",
        help="Comma-separated fields used as answer text, e.g. 'completion' or 'output'.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Max rows/groups to print for each issue type.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=90,
        help="Preview length used in report output.",
    )
    parser.add_argument(
        "--no-collapse-whitespace",
        action="store_true",
        help="Disable whitespace collapsing when checking duplicate questions.",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default="",
        help="Optional output JSON report path.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1

    try:
        items = load_json_items(input_path)
    except Exception as exc:
        print(f"[ERROR] Failed to read JSON: {exc}")
        return 1

    sample_item = next((item for item in items if isinstance(item, dict)), None)
    if sample_item is None:
        print("[ERROR] JSON list does not contain dict records.")
        return 1

    question_fields = parse_fields_arg(args.question_fields)
    answer_fields = parse_fields_arg(args.answer_fields)

    if not question_fields:
        question_fields = detect_field_combo(sample_item, QUESTION_FIELD_PRIORITY)
    if not answer_fields:
        answer_fields = detect_field_combo(sample_item, ANSWER_FIELD_PRIORITY)

    if not question_fields:
        print("[ERROR] Could not detect question fields. Please set --question-fields.")
        return 1
    if not answer_fields:
        print("[ERROR] Could not detect answer fields. Please set --answer-fields.")
        return 1

    report = analyze_file(
        items=items,
        question_fields=question_fields,
        answer_fields=answer_fields,
        preview_chars=max(1, args.preview_chars),
        collapse_ws_for_duplicate=not args.no_collapse_whitespace,
    )

    print(f"[FILE] {input_path}")
    print_report(report, max_examples=max(0, args.max_examples))

    if args.report_file:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        full_report = {
            "input_file": str(input_path),
            "report": report,
        }
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)
        print(f"\n[REPORT_SAVED] {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
