from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_INPUT_CANDIDATES = [
    Path("data/processed/gb2760_a1_grouped_min.jsonl"),
    Path("data/processed/gb2760_a1_grouped.jsonl"),
]
DEFAULT_JSONL_OUTPUT = Path("data/processed/gb2760_a1_grouped_min_final.jsonl")
DEFAULT_PRETTY_OUTPUT = Path("data/processed/gb2760_a1_grouped_min_final_pretty.json")
DEFAULT_REPORT_OUTPUT = Path("data/processed/gb2760_a1_grouped_min_final_report.md")
DEFAULT_DEBUG_OUTPUT = Path("data/processed/gb2760_a1_grouped_min_final_debug.json")
ID_PREFIX = "GB2760_A1_TERM"
STANDARD_NO = "GB 2760-2024"
TABLE_NO = "A.1"

ALIAS_MARKER_RE = re.compile(r"^(?:又名|别名|简称|俗称|常用名|商品名|又称|也称|亦称|通称)")
TRAILING_CODE_RE = re.compile(r"\s*(?:CNS|INS)号\s*[-A-Za-z0-9.,]+\s*$")
WHITESPACE_RE = re.compile(r"\s+")
NON_ALIAS_EXACT = {
    "液",
    "乳液",
    "合成",
    "普通法",
    "加氨生产",
    "亚硫酸铵法",
    "煤气化法",
    "红",
    "黑",
    "白",
    "黄",
    "蓝",
    "绿",
}
NON_ALIAS_SUBSTRINGS = (
    "仅限",
    "除外",
    "不包括",
    "包括",
    "工艺",
    "适用",
    "范围",
    "产品",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize GB2760 A.1 grouped_min JSONL for vector DB ingestion."
    )
    parser.add_argument("--input", default="", help="Input grouped_min JSONL path.")
    parser.add_argument(
        "--jsonl-output",
        default=str(DEFAULT_JSONL_OUTPUT),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--pretty-output",
        default=str(DEFAULT_PRETTY_OUTPUT),
        help="Output pretty JSON path.",
    )
    parser.add_argument(
        "--report-output",
        default=str(DEFAULT_REPORT_OUTPUT),
        help="Output report Markdown path.",
    )
    parser.add_argument(
        "--debug-output",
        default=str(DEFAULT_DEBUG_OUTPUT),
        help="Output debug JSON path.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u3000", " ").replace("\xa0", " ").replace("\r", " ").replace("\n", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = re.sub(r"\s+([，。；：、）】》〉])", r"\1", text)
    text = re.sub(r"([（【《〈])\s+", r"\1", text)
    return text


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = normalize_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def parse_code_sort_key(code: str) -> tuple[Any, ...]:
    parts: list[tuple[int, Any]] = []
    for piece in str(code).strip().split("."):
        if piece.isdigit():
            parts.append((0, int(piece)))
        else:
            parts.append((1, piece))
    return tuple(parts)


def resolve_input_path(explicit_input: str) -> Path:
    if explicit_input:
        path = Path(explicit_input)
        if path.exists() and path.is_file():
            return path
        raise FileNotFoundError(f"Input file not found: {path}")

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists() and candidate.is_file():
            return candidate

    ranked: list[tuple[int, Path]] = []
    for path in Path(".").rglob("*.jsonl"):
        lower = path.name.lower()
        if "final" in lower:
            continue
        score = 0
        if "gb2760" in lower:
            score += 10
        if "a1" in lower:
            score += 10
        if "grouped_min" in lower:
            score += 20
        elif "grouped" in lower:
            score += 8
        if "processed" in str(path.parent).lower():
            score += 5
        if score > 0:
            ranked.append((score, path))

    if not ranked:
        raise FileNotFoundError("No suitable grouped_min JSONL input file was found.")

    ranked.sort(key=lambda item: (-item[0], len(str(item[1])), str(item[1])))
    return ranked[0][1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            records.append(obj)
    return records


def clean_term_text(term: str) -> str:
    cleaned = normalize_text(term)
    cleaned = TRAILING_CODE_RE.sub("", cleaned)
    cleaned = normalize_text(cleaned).strip("，,；; ")
    return cleaned


def find_parenthetical_segments(text: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    stack: list[tuple[str, int]] = []
    pairs = {"（": "）", "(": ")"}
    reverse_pairs = {value: key for key, value in pairs.items()}
    for index, char in enumerate(text):
        if char in pairs:
            stack.append((char, index))
        elif char in reverse_pairs and stack:
            open_char, start = stack.pop()
            if pairs[open_char] == char and not stack:
                segments.append(
                    {
                        "start": start,
                        "end": index + 1,
                        "content": text[start + 1 : index],
                    }
                )
    return segments


def split_alias_candidates(content: str) -> list[str]:
    stripped = normalize_text(content)
    stripped = ALIAS_MARKER_RE.sub("", stripped).strip("：: ")
    parts = re.split(r"[，,；;、/]+", stripped)
    return [normalize_text(part) for part in parts if normalize_text(part)]


def looks_like_acronym(content: str) -> bool:
    compact = normalize_text(content).replace(" ", "")
    if not compact or len(compact) > 16:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9+\-α-ωΑ-Ω]+", compact))


def is_non_alias_content(content: str) -> bool:
    compact = normalize_text(content)
    if not compact:
        return True
    if compact in NON_ALIAS_EXACT:
        return True
    if any(token in compact for token in NON_ALIAS_SUBSTRINGS):
        return True
    if re.search(r"\d{2}\.\d{2}", compact):
        return True
    if compact.endswith("法") and len(compact) <= 8:
        return True
    if compact.endswith("生产") and len(compact) <= 8:
        return True
    return False


def classify_parenthetical(content: str, before_text: str, after_text: str, full_term: str) -> tuple[bool, str]:
    compact = normalize_text(content)
    if not compact:
        return False, "empty"
    if ALIAS_MARKER_RE.match(compact):
        return True, "marker"
    if is_non_alias_content(compact):
        return False, "descriptor"

    before_tail = normalize_text(before_text).rstrip()
    after_head = normalize_text(after_text).lstrip()
    if before_tail.endswith(("-", "－")) or after_head.startswith(("-", "－")):
        return False, "structural"

    if looks_like_acronym(compact):
        return True, "acronym"

    parts = split_alias_candidates(compact)
    if len(parts) > 4:
        return False, "enumeration"

    max_len = max((len(part) for part in parts), default=0)
    if len(parts) == 1 and max_len <= 8:
        return True, "short"
    if 1 < len(parts) <= 3 and max_len <= 10 and "及其" not in compact:
        return True, "short_list"
    if "维生素" in full_term and len(parts) <= 4 and max_len <= 14:
        return True, "vitamin_alias"

    return False, "keep"


def rebuild_term_fields(raw_term: str) -> dict[str, Any]:
    cleaned_term = clean_term_text(raw_term)
    segments = find_parenthetical_segments(cleaned_term)
    aliases: list[str] = []
    alias_reasons: list[str] = []
    manual_review_reasons: list[str] = []

    pieces: list[str] = []
    cursor = 0
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        content = normalize_text(segment["content"])
        before = cleaned_term[cursor:start]
        after = cleaned_term[end:]
        is_alias, reason = classify_parenthetical(content, cleaned_term[:start], after, cleaned_term)
        pieces.append(cleaned_term[cursor:start])
        if is_alias:
            aliases.extend(split_alias_candidates(content))
            alias_reasons.append(reason)
            if reason not in {"marker", "acronym"}:
                manual_review_reasons.append(f"heuristic_alias:{reason}")
        else:
            pieces.append(cleaned_term[start:end])
            if reason not in {"descriptor", "structural", "enumeration", "keep", "empty"}:
                manual_review_reasons.append(f"kept_parenthetical:{reason}")
        cursor = end
    pieces.append(cleaned_term[cursor:])

    normalized_term = "".join(pieces)
    normalized_term = normalize_text(normalized_term)
    normalized_term = re.sub(r"[，,]{2,}", "，", normalized_term)
    normalized_term = re.sub(r"\s*([，,；;])\s*", r"\1", normalized_term)
    normalized_term = normalized_term.strip("，,；; ")
    if not normalized_term:
        normalized_term = cleaned_term

    aliases = dedupe_preserve_order(aliases)
    if "（" in cleaned_term or "(" in cleaned_term:
        if aliases:
            pass
        else:
            manual_review_reasons.append("parenthetical_without_alias")
    if cleaned_term != raw_term:
        manual_review_reasons.append("term_cleaned")

    return {
        "term": cleaned_term,
        "normalized_term": normalized_term,
        "aliases": aliases,
        "alias_reasons": alias_reasons,
        "manual_review_reasons": dedupe_preserve_order(manual_review_reasons),
    }


def normalize_rule(rule: dict[str, Any]) -> dict[str, str]:
    return {
        "food_category_code": normalize_text(rule.get("food_category_code", "")),
        "food_category_name": normalize_text(rule.get("food_category_name", "")),
        "usage_limit": normalize_text(rule.get("usage_limit", "")),
        "remarks": normalize_text(rule.get("remarks", "")),
    }


def dedupe_rules(rules: list[dict[str, Any]]) -> tuple[list[dict[str, str]], int]:
    normalized = [normalize_rule(rule) for rule in rules]
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    removed = 0
    for rule in normalized:
        key = (
            rule["food_category_code"],
            rule["food_category_name"],
            rule["usage_limit"],
            rule["remarks"],
        )
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique.append(rule)
    unique.sort(key=lambda item: parse_code_sort_key(item["food_category_code"]))
    return unique, removed


def build_keywords(term: str, normalized_term: str, aliases: list[str], function_category: str, rules: list[dict[str, str]]) -> list[str]:
    food_names = dedupe_preserve_order([rule["food_category_name"] for rule in rules])[:6]
    return dedupe_preserve_order([normalized_term, term, *aliases, function_category, "GB2760", *food_names])


def build_embedding_text(term: str, aliases: list[str], function_category: str, rules: list[dict[str, str]]) -> str:
    food_names = dedupe_preserve_order([rule["food_category_name"] for rule in rules])[:10]
    usage_limits = dedupe_preserve_order([rule["usage_limit"] for rule in rules if rule["usage_limit"]])[:6]
    remarks = dedupe_preserve_order([rule["remarks"] for rule in rules if rule["remarks"]])[:3]

    text = (
        f"GB 2760-2024 A.1 规定，食品添加剂{term}属于{function_category}，"
    )
    if aliases:
        text += f"常用名包括{'、'.join(aliases)}，"
    text += f"可用于{'、'.join(food_names)}等食品。"
    if usage_limits:
        text += f"典型使用限量包括 {'、'.join(usage_limits)}。"
    if remarks:
        text += f"备注包括：{'；'.join(remarks)}。"
    return text


def build_id(normalized_term: str) -> str:
    seed = f"{STANDARD_NO}|{TABLE_NO}|{normalized_term}"
    short_hash = hashlib.md5(seed.encode("utf-8")).hexdigest()[:8]
    return f"{ID_PREFIX}_{short_hash}"


def finalize_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    aliases_restored_records = 0
    normalized_term_fixed_records = 0
    rules_dedup_removed = 0
    aliases_empty_records = 0
    id_changed_records = 0
    manual_review_examples: list[dict[str, Any]] = []
    sample_diffs: list[dict[str, Any]] = []

    for record in records:
        original_term = normalize_text(record.get("term", ""))
        original_normalized = normalize_text(record.get("normalized_term", ""))
        function_category = normalize_text(record.get("function_category", ""))
        rebuilt = rebuild_term_fields(original_term)
        cleaned_rules, removed_count = dedupe_rules(list(record.get("rules") or []))
        rules_dedup_removed += removed_count

        final_id = build_id(rebuilt["normalized_term"])
        if final_id != normalize_text(record.get("id", "")):
            id_changed_records += 1

        keywords = build_keywords(
            term=rebuilt["term"],
            normalized_term=rebuilt["normalized_term"],
            aliases=rebuilt["aliases"],
            function_category=function_category,
            rules=cleaned_rules,
        )
        embedding_text = build_embedding_text(
            term=rebuilt["term"],
            aliases=rebuilt["aliases"],
            function_category=function_category,
            rules=cleaned_rules,
        )

        final_obj = {
            "id": final_id,
            "term": rebuilt["term"],
            "normalized_term": rebuilt["normalized_term"],
            "aliases": rebuilt["aliases"],
            "function_category": function_category,
            "rules": cleaned_rules,
            "keywords": keywords,
            "embedding_text": embedding_text,
        }
        finalized.append(final_obj)

        if rebuilt["aliases"]:
            aliases_restored_records += 1
        else:
            aliases_empty_records += 1
        if rebuilt["normalized_term"] != original_normalized:
            normalized_term_fixed_records += 1

        if rebuilt["manual_review_reasons"] and len(manual_review_examples) < 12:
            manual_review_examples.append(
                {
                    "term_before": original_term,
                    "term_after": rebuilt["term"],
                    "normalized_before": original_normalized,
                    "normalized_after": rebuilt["normalized_term"],
                    "aliases": rebuilt["aliases"],
                    "reasons": rebuilt["manual_review_reasons"],
                }
            )

        if len(sample_diffs) < 3 and (
            rebuilt["aliases"]
            or rebuilt["normalized_term"] != original_normalized
            or removed_count
            or rebuilt["term"] != original_term
        ):
            sample_diffs.append(
                {
                    "term_before": original_term,
                    "term_after": rebuilt["term"],
                    "normalized_before": original_normalized,
                    "normalized_after": rebuilt["normalized_term"],
                    "aliases": rebuilt["aliases"],
                    "rules_before": len(list(record.get("rules") or [])),
                    "rules_after": len(cleaned_rules),
                }
            )

    finalized.sort(key=lambda item: (item["term"], item["function_category"], item["id"]))
    stats = {
        "total_records": len(finalized),
        "aliases_restored_records": aliases_restored_records,
        "normalized_term_fixed_records": normalized_term_fixed_records,
        "rules_dedup_removed": rules_dedup_removed,
        "aliases_empty_records": aliases_empty_records,
        "id_changed_records": id_changed_records,
        "manual_review_examples": manual_review_examples,
        "sample_diffs": sample_diffs,
    }
    return finalized, stats


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_pretty_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def write_debug_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_report(
    input_path: Path,
    jsonl_output: Path,
    pretty_output: Path,
    report_output: Path,
    debug_output: Path,
    stats: dict[str, Any],
) -> str:
    sample_diffs = stats.get("sample_diffs") or []
    if not sample_diffs:
        sample_section = "- 无明显变化样本\n"
    else:
        sample_lines = []
        for sample in sample_diffs[:3]:
            sample_lines.append(
                "- `{before}` -> `{after}`；normalized_term: `{nb}` -> `{na}`；aliases: `{aliases}`；rules: `{rb}` -> `{ra}`".format(
                    before=sample["term_before"],
                    after=sample["term_after"],
                    nb=sample["normalized_before"],
                    na=sample["normalized_after"],
                    aliases=", ".join(sample["aliases"]) if sample["aliases"] else "[]",
                    rb=sample["rules_before"],
                    ra=sample["rules_after"],
                )
            )
        sample_section = "\n".join(sample_lines) + "\n"

    return (
        "# GB2760 A.1 Finalize Report\n\n"
        f"- 输入文件路径：`{input_path.resolve()}`\n"
        f"- JSONL 输出路径：`{jsonl_output.resolve()}`\n"
        f"- Pretty JSON 输出路径：`{pretty_output.resolve()}`\n"
        f"- Report 输出路径：`{report_output.resolve()}`\n"
        f"- Debug JSON 输出路径：`{debug_output.resolve()}`\n"
        f"- 总记录数：`{stats['total_records']}`\n"
        f"- aliases 恢复数量：`{stats['aliases_restored_records']}`\n"
        f"- normalized_term 修正数量：`{stats['normalized_term_fixed_records']}`\n"
        f"- rules 去重数量：`{stats['rules_dedup_removed']}`\n\n"
        "## 抽样 3 条记录的处理前后对比摘要\n"
        f"{sample_section}"
    )


def main() -> int:
    args = parse_args()
    try:
        input_path = resolve_input_path(args.input)
        jsonl_output = Path(args.jsonl_output)
        pretty_output = Path(args.pretty_output)
        report_output = Path(args.report_output)
        debug_output = Path(args.debug_output)

        records = load_jsonl(input_path)
        finalized, stats = finalize_records(records)

        write_jsonl(jsonl_output, finalized)
        write_pretty_json(pretty_output, finalized)
        debug_payload = {
            "input_file": str(input_path.resolve()),
            "jsonl_output": str(jsonl_output.resolve()),
            "pretty_output": str(pretty_output.resolve()),
            "report_output": str(report_output.resolve()),
            "debug_output": str(debug_output.resolve()),
            "total_records": stats["total_records"],
            "aliases_restored_records": stats["aliases_restored_records"],
            "normalized_term_fixed_records": stats["normalized_term_fixed_records"],
            "rules_dedup_removed": stats["rules_dedup_removed"],
            "aliases_empty_records": stats["aliases_empty_records"],
            "id_changed_records": stats["id_changed_records"],
            "manual_review_examples": stats["manual_review_examples"],
        }
        write_debug_json(debug_output, debug_payload)
        report_text = build_report(
            input_path=input_path,
            jsonl_output=jsonl_output,
            pretty_output=pretty_output,
            report_output=report_output,
            debug_output=debug_output,
            stats=stats,
        )
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(report_text, encoding="utf-8")

        summary = {
            "input": str(input_path.resolve()),
            "jsonl_output": str(jsonl_output.resolve()),
            "pretty_output": str(pretty_output.resolve()),
            "report_output": str(report_output.resolve()),
            "debug_output": str(debug_output.resolve()),
            "total_records": stats["total_records"],
            "aliases_restored_records": stats["aliases_restored_records"],
            "normalized_term_fixed_records": stats["normalized_term_fixed_records"],
            "rules_dedup_removed": stats["rules_dedup_removed"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
