from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

STANDARD_NO = "GB 2760-2024"
TABLE_NO = "A.1"
ID_PREFIX = "GB2760_A1_TERM"
DEFAULT_INPUT_CANDIDATES = [
    Path("data/processed/gb2760_a1_grouped_min_final.jsonl"),
    Path("data/processed/gb2760_a1_grouped_min.jsonl"),
]
DEFAULT_JSONL_OUTPUT = Path("data/processed/gb2760_a1_grouped_min_final_v2.jsonl")
DEFAULT_PRETTY_OUTPUT = Path("data/processed/gb2760_a1_grouped_min_final_v2_pretty.json")
DEFAULT_REPORT_OUTPUT = Path("data/processed/gb2760_a1_grouped_min_final_v2_report.md")
DEFAULT_DEBUG_OUTPUT = Path("data/processed/gb2760_a1_grouped_min_final_v2_debug.json")
ALIAS_MARKER_RE = re.compile(r"^(?:又名|别名|简称|俗称|常用名|商品名|又称|也称|亦称|通称)")
DESCRIPTOR_TOKENS = ("仅限", "除外", "不包括", "包括", "工艺", "生产", "法", "型")
DESCRIPTOR_EXACT = {"液", "乳液", "合成", "红", "黑", "白", "黄", "蓝", "绿", "煤气化法", "普通法", "加氨生产", "亚硫酸铵法"}
SHORT_NON_SPLIT_SEGMENTS = {"单", "双", "三", "四", "五", "六", "七", "八", "九", "十"}
CHEMICAL_TAIL_TOKENS = ("酸", "盐", "酯", "胶", "素", "淀", "剂", "醇", "酚", "糖", "氧", "氢", "钠", "钾", "钙")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize GB2760 A.1 grouped_min JSONL into v2 outputs.")
    parser.add_argument("--input", default="", help="Input grouped_min_final JSONL path.")
    parser.add_argument("--jsonl-output", default=str(DEFAULT_JSONL_OUTPUT), help="Output JSONL path.")
    parser.add_argument("--pretty-output", default=str(DEFAULT_PRETTY_OUTPUT), help="Output pretty JSON path.")
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_OUTPUT), help="Output report Markdown path.")
    parser.add_argument("--debug-output", default=str(DEFAULT_DEBUG_OUTPUT), help="Output debug JSON path.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u3000", " ").replace("\xa0", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
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
        if "gb2760" not in lower or "a1" not in lower or "grouped_min_final" not in lower or "v2" in lower:
            continue
        score = 0
        if "processed" in str(path.parent).lower():
            score += 5
        if lower.endswith("grouped_min_final.jsonl"):
            score += 10
        ranked.append((score, path))
    if not ranked:
        raise FileNotFoundError("No suitable grouped_min_final JSONL file was found.")
    ranked.sort(key=lambda item: (-item[0], len(str(item[1])), str(item[1])))
    return ranked[0][1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(obj)
    return rows


def parse_code_sort_key(code: str) -> tuple[Any, ...]:
    parts: list[tuple[int, Any]] = []
    for piece in str(code).strip().split("."):
        if piece.isdigit():
            parts.append((0, int(piece)))
        else:
            parts.append((1, piece))
    return tuple(parts)


def split_top_level(text: str, delimiters: set[str]) -> list[str]:
    parts: list[str] = []
    stack: list[str] = []
    start = 0
    pairs = {"（": "）", "(": ")"}
    reverse_pairs = {value: key for key, value in pairs.items()}
    for idx, char in enumerate(text):
        if char in pairs:
            stack.append(char)
        elif char in reverse_pairs and stack and stack[-1] == reverse_pairs[char]:
            stack.pop()
        elif char in delimiters and not stack:
            parts.append(text[start:idx])
            start = idx + 1
    parts.append(text[start:])
    return [normalize_text(part) for part in parts if normalize_text(part)]


def normalize_parallel_punctuation(text: str) -> str:
    chars: list[str] = []
    stack: list[str] = []
    pairs = {"（": "）", "(": ")"}
    reverse_pairs = {value: key for key, value in pairs.items()}
    for char in text:
        if char in pairs:
            stack.append(char)
            chars.append(char)
        elif char in reverse_pairs and stack and stack[-1] == reverse_pairs[char]:
            stack.pop()
            chars.append(char)
        elif char in {"，", "；"}:
            chars.append("、")
        else:
            chars.append(char)
    return "".join(chars)


def find_parenthetical_segments(text: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    stack: list[tuple[str, int]] = []
    pairs = {"（": "）", "(": ")"}
    reverse_pairs = {value: key for key, value in pairs.items()}
    for idx, char in enumerate(text):
        if char in pairs:
            stack.append((char, idx))
        elif char in reverse_pairs and stack:
            open_char, start = stack.pop()
            if pairs[open_char] == char and not stack:
                segments.append({"start": start, "end": idx + 1, "content": text[start + 1 : idx]})
    return segments


def split_alias_candidates(content: str) -> list[str]:
    stripped = ALIAS_MARKER_RE.sub("", normalize_text(content)).strip("：: ")
    parts = re.split(r"[、，,；;/]+", stripped)
    return [normalize_text(part) for part in parts if normalize_text(part)]


def looks_like_alias(content: str) -> bool:
    compact = normalize_text(content).replace(" ", "")
    if not compact:
        return False
    if ALIAS_MARKER_RE.match(compact):
        return True
    if compact in DESCRIPTOR_EXACT:
        return False
    if any(token in compact for token in DESCRIPTOR_TOKENS):
        return False
    if re.fullmatch(r"[A-Za-z0-9+\-α-ωΑ-Ω]{2,16}", compact):
        return True
    if len(compact) <= 8:
        return True
    return False


def looks_like_structural_list(content: str) -> bool:
    compact = normalize_text(content)
    if not compact or any(token in compact for token in DESCRIPTOR_TOKENS):
        return False
    parts = split_top_level(normalize_parallel_punctuation(compact), {"、"})
    if len(parts) < 2:
        return False
    if any(len(part) < 2 for part in parts):
        return False
    if sum(any(token in part for token in CHEMICAL_TAIL_TOKENS) for part in parts) >= 2:
        return True
    if len(parts) >= 4:
        return True
    return False


def rebuild_normalized_term(term: str, existing_aliases: list[str]) -> tuple[str, list[str], bool, list[str]]:
    clean_term = normalize_text(term)
    clean_term = re.sub(r"\s+([）】》〉])", r"\1", clean_term)
    clean_term = re.sub(r"([（【《〈])\s+", r"\1", clean_term)
    clean_term = re.sub(r"\s*([，、；：])\s*", r"\1", clean_term)
    clean_term = normalize_parallel_punctuation(clean_term)
    aliases = list(existing_aliases or [])
    changed = False
    reasons: list[str] = []

    pieces: list[str] = []
    cursor = 0
    for segment in find_parenthetical_segments(clean_term):
        start = segment["start"]
        end = segment["end"]
        content = normalize_text(segment["content"])
        before_tail = clean_term[:start]
        after_text = clean_term[end:]
        pieces.append(clean_term[cursor:start])

        structural = before_tail.rstrip().endswith(("-", "－")) or after_text.lstrip().startswith(("-", "－"))
        if structural:
            pieces.append(clean_term[start:end])
        elif looks_like_alias(content):
            aliases.extend(split_alias_candidates(content))
            changed = True
            reasons.append(f"drop_alias_paren:{content}")
        elif looks_like_structural_list(content):
            changed = True
            reasons.append(f"drop_structural_list:{content[:50]}")
        else:
            pieces.append(clean_term[start:end])
        cursor = end
    pieces.append(clean_term[cursor:])

    normalized_term = normalize_text("".join(pieces))
    normalized_term = re.sub(r"\s*([，、；：])\s*", r"\1", normalized_term)
    normalized_term = normalize_parallel_punctuation(normalized_term)
    normalized_term = re.sub(r"、+", "、", normalized_term).strip("、")
    normalized_term = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", normalized_term)
    if normalized_term != normalize_text(term):
        changed = True
    return normalized_term, dedupe_preserve_order(aliases), changed, dedupe_preserve_order(reasons)


def clean_food_category_name(name: str) -> tuple[str, bool]:
    original = normalize_text(name)
    cleaned = original
    cleaned = re.sub(r"\s*([（(])\s*", r"\1", cleaned)
    cleaned = re.sub(r"\s*([）)])\s*", r"\1", cleaned)
    cleaned = re.sub(r"\s*([，、；：])\s*", r"\1", cleaned)
    cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", cleaned)
    cleaned = re.sub(r"(?<=\d)\s+(?=[\u4e00-\u9fff])", "", cleaned)
    cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=\d)", "", cleaned)
    cleaned = normalize_text(cleaned)
    return cleaned, cleaned != original


def dedupe_rules_and_check_conflicts(rules: list[dict[str, Any]]) -> tuple[list[dict[str, str]], int, list[dict[str, Any]], int]:
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    dedup_removed = 0
    cleaned_name_changes = 0
    conflict_map: dict[tuple[str, str], dict[str, Any]] = {}

    for raw_rule in rules:
        cleaned_name, changed = clean_food_category_name(raw_rule.get("food_category_name", ""))
        if changed:
            cleaned_name_changes += 1
        rule = {
            "food_category_code": normalize_text(raw_rule.get("food_category_code", "")),
            "food_category_name": cleaned_name,
            "usage_limit": normalize_text(raw_rule.get("usage_limit", "")),
            "remarks": normalize_text(raw_rule.get("remarks", "")),
        }
        key = (rule["food_category_code"], rule["food_category_name"], rule["usage_limit"], rule["remarks"])
        if key in seen:
            dedup_removed += 1
            continue
        seen.add(key)
        unique.append(rule)

        conflict_key = (rule["food_category_code"], rule["food_category_name"])
        bucket = conflict_map.setdefault(conflict_key, {"usage_limits": set(), "remarks": set()})
        bucket["usage_limits"].add(rule["usage_limit"])
        if rule["remarks"]:
            bucket["remarks"].add(rule["remarks"])

    unique.sort(key=lambda item: parse_code_sort_key(item["food_category_code"]))
    conflicts: list[dict[str, Any]] = []
    for (code, name), bucket in conflict_map.items():
        if len(bucket["usage_limits"]) > 1:
            conflicts.append(
                {
                    "food_category_code": code,
                    "food_category_name": name,
                    "usage_limits": sorted(bucket["usage_limits"]),
                    "remarks": sorted(bucket["remarks"]),
                }
            )
    return unique, dedup_removed, conflicts, cleaned_name_changes


def split_parallel_keyword_candidates(text: str) -> list[str]:
    compact = normalize_text(text)
    if "、" not in compact:
        return []
    parts = split_top_level(compact, {"、"})
    if len(parts) < 2 or any(part in SHORT_NON_SPLIT_SEGMENTS or len(part) < 2 for part in parts):
        return []
    return parts


def rebuild_keywords(normalized_term: str, term: str, aliases: list[str], function_category: str, rules: list[dict[str, str]]) -> tuple[list[str], bool]:
    representative_foods = dedupe_preserve_order([rule["food_category_name"] for rule in rules])[:6]
    base = [normalized_term, term, *aliases, function_category, "GB2760", *representative_foods]
    extra_parallel: list[str] = []
    extra_parallel.extend(split_parallel_keyword_candidates(normalized_term))
    for alias in aliases:
        extra_parallel.extend(split_parallel_keyword_candidates(alias))
    keywords = dedupe_preserve_order(base + extra_parallel)
    return keywords, bool(dedupe_preserve_order(extra_parallel))


def summarize_items(values: list[str], limit: int, max_chars: int | None = None) -> list[str]:
    unique = dedupe_preserve_order(values)
    result: list[str] = []
    for value in unique:
        item = value
        if max_chars and len(item) > max_chars:
            item = item[: max_chars - 1].rstrip() + "…"
        result.append(item)
        if len(result) >= limit:
            break
    return result


def rebuild_embedding_text(term: str, aliases: list[str], function_category: str, rules: list[dict[str, str]]) -> str:
    food_names = summarize_items([rule["food_category_name"] for rule in rules], limit=8, max_chars=24)
    usage_limits = summarize_items([rule["usage_limit"] for rule in rules if rule["usage_limit"]], limit=4)
    remarks = summarize_items([rule["remarks"] for rule in rules if rule["remarks"]], limit=2, max_chars=28)

    text = f"GB 2760-2024 A.1 规定，食品添加剂{term}属于{function_category}。"
    if aliases:
        text += f"别名包括{'、'.join(aliases[:4])}。"
    if food_names:
        text += f"可用于{'、'.join(food_names)}等食品。"
    if usage_limits:
        text += f"典型使用限量包括 {'、'.join(usage_limits)}。"
    if remarks:
        text += f"备注涉及{'；'.join(remarks)}。"
    return text


def build_id(normalized_term: str) -> str:
    seed = f"{STANDARD_NO}|{TABLE_NO}|{normalized_term}"
    short_hash = hashlib.md5(seed.encode("utf-8")).hexdigest()[:8]
    return f"{ID_PREFIX}_{short_hash}"


def process_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    normalized_term_changed_count = 0
    food_category_name_cleaned_count = 0
    embedding_shortened_count = 0
    grouped_conflict_count = 0
    parallel_keyword_records = 0
    exact_rules_dedup_removed = 0
    conflict_examples: list[dict[str, Any]] = []
    normalized_examples: list[dict[str, Any]] = []
    category_examples: list[dict[str, Any]] = []

    for record in records:
        term = normalize_text(record.get("term", ""))
        old_normalized = normalize_text(record.get("normalized_term", ""))
        aliases = [normalize_text(alias) for alias in (record.get("aliases") or []) if normalize_text(alias)]
        function_category = normalize_text(record.get("function_category", ""))
        old_embedding = normalize_text(record.get("embedding_text", ""))

        new_normalized, new_aliases, normalized_changed, reasons = rebuild_normalized_term(term, aliases)
        actual_normalized_changed = new_normalized != old_normalized
        if actual_normalized_changed:
            normalized_term_changed_count += 1
            if len(normalized_examples) < 12:
                normalized_examples.append(
                    {
                        "term": term,
                        "normalized_before": old_normalized,
                        "normalized_after": new_normalized,
                        "reasons": reasons,
                    }
                )

        clean_rules, dedup_removed, conflicts, cleaned_rule_name_changes = dedupe_rules_and_check_conflicts(list(record.get("rules") or []))
        exact_rules_dedup_removed += dedup_removed
        food_category_name_cleaned_count += cleaned_rule_name_changes
        if cleaned_rule_name_changes and len(category_examples) < 12:
            for raw_rule, clean_rule in zip(list(record.get("rules") or []), clean_rules):
                before_name = normalize_text(raw_rule.get("food_category_name", ""))
                after_name = clean_rule["food_category_name"]
                if before_name != after_name:
                    category_examples.append(
                        {
                            "term": term,
                            "food_category_name_before": before_name,
                            "food_category_name_after": after_name,
                        }
                    )
                    break

        if conflicts:
            grouped_conflict_count += len(conflicts)
            if len(conflict_examples) < 12:
                conflict_examples.append(
                    {
                        "term": term,
                        "normalized_term": new_normalized,
                        "conflicts": conflicts,
                    }
                )

        keywords, has_parallel_keywords = rebuild_keywords(new_normalized, term, new_aliases, function_category, clean_rules)
        if has_parallel_keywords:
            parallel_keyword_records += 1

        new_embedding = rebuild_embedding_text(term, new_aliases, function_category, clean_rules)
        if old_embedding and len(new_embedding) < len(old_embedding):
            embedding_shortened_count += 1

        outputs.append(
            {
                "id": build_id(new_normalized),
                "term": term,
                "normalized_term": new_normalized,
                "aliases": new_aliases,
                "function_category": function_category,
                "rules": clean_rules,
                "keywords": keywords,
                "embedding_text": new_embedding,
            }
        )

    outputs.sort(key=lambda item: (item["term"], item["function_category"], item["id"]))
    stats = {
        "total_records": len(outputs),
        "normalized_term_changed_count": normalized_term_changed_count,
        "food_category_name_cleaned_count": food_category_name_cleaned_count,
        "embedding_shortened_count": embedding_shortened_count,
        "grouped_conflict_count": grouped_conflict_count,
        "parallel_keyword_records": parallel_keyword_records,
        "exact_rules_dedup_removed": exact_rules_dedup_removed,
        "normalized_examples": normalized_examples,
        "category_examples": category_examples,
        "conflict_examples": conflict_examples,
    }
    return outputs, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_pretty_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def write_debug_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_report(input_path: Path, outputs: dict[str, Path], stats: dict[str, Any]) -> str:
    lines = [
        "# GB2760 A.1 Final V2 Report",
        "",
        f"- 输入文件路径：`{input_path.resolve()}`",
        f"- JSONL 输出路径：`{outputs['jsonl'].resolve()}`",
        f"- Pretty JSON 输出路径：`{outputs['pretty'].resolve()}`",
        f"- Report 输出路径：`{outputs['report'].resolve()}`",
        f"- Debug JSON 输出路径：`{outputs['debug'].resolve()}`",
        f"- 总记录数：`{stats['total_records']}`",
        f"- 修正 normalized_term 数量：`{stats['normalized_term_changed_count']}`",
        f"- 清洗 food_category_name 数量：`{stats['food_category_name_cleaned_count']}`",
        f"- 缩短 embedding_text 数量：`{stats['embedding_shortened_count']}`",
        f"- 检测到的 grouped 内部冲突数量：`{stats['grouped_conflict_count']}`",
        f"- 补充并列关键词的记录数：`{stats['parallel_keyword_records']}`",
        f"- exact 重复 rules 去重数量：`{stats['exact_rules_dedup_removed']}`",
        "",
        "## normalized_term 修正样例",
    ]
    if stats["normalized_examples"]:
        for item in stats["normalized_examples"][:3]:
            lines.append(
                f"- `{item['term']}`: `{item['normalized_before']}` -> `{item['normalized_after']}`"
            )
    else:
        lines.append("- 无")

    lines.append("")
    lines.append("## grouped 内部冲突样例")
    if stats["conflict_examples"]:
        for item in stats["conflict_examples"][:3]:
            lines.append(f"- `{item['term']}`: `{json.dumps(item['conflicts'], ensure_ascii=False)}`")
    else:
        lines.append("- 未检测到")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        input_path = resolve_input_path(args.input)
        jsonl_output = Path(args.jsonl_output)
        pretty_output = Path(args.pretty_output)
        report_output = Path(args.report_output)
        debug_output = Path(args.debug_output)

        records = load_jsonl(input_path)
        outputs, stats = process_records(records)
        write_jsonl(jsonl_output, outputs)
        write_pretty_json(pretty_output, outputs)
        debug_payload = {
            "input_file": str(input_path.resolve()),
            "jsonl_output": str(jsonl_output.resolve()),
            "pretty_output": str(pretty_output.resolve()),
            "report_output": str(report_output.resolve()),
            "debug_output": str(debug_output.resolve()),
            **stats,
        }
        write_debug_json(debug_output, debug_payload)
        report_text = build_report(
            input_path,
            {"jsonl": jsonl_output, "pretty": pretty_output, "report": report_output, "debug": debug_output},
            stats,
        )
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(report_text, encoding="utf-8")

        summary = {
            "total_records": stats["total_records"],
            "normalized_term_changed_count": stats["normalized_term_changed_count"],
            "food_category_name_cleaned_count": stats["food_category_name_cleaned_count"],
            "embedding_shortened_count": stats["embedding_shortened_count"],
            "grouped_conflict_count": stats["grouped_conflict_count"],
            "parallel_keyword_records": stats["parallel_keyword_records"],
            "jsonl_output": str(jsonl_output.resolve()),
            "pretty_output": str(pretty_output.resolve()),
            "report_output": str(report_output.resolve()),
            "debug_output": str(debug_output.resolve()),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
