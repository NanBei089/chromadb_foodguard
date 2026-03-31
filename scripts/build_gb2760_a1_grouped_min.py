from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_INPUT_CANDIDATES = [
    Path("data/processed/gb2760_a1_rules_ready.jsonl"),
    Path("data/processed/gb2760_a1_rules.jsonl"),
    Path("output/gb2760_a1_rules.jsonl"),
]
DEFAULT_JSONL_OUTPUT = Path("data/processed/gb2760_a1_grouped_min.jsonl")
DEFAULT_PRETTY_OUTPUT = Path("data/processed/gb2760_a1_grouped_pretty.json")
DEFAULT_REPORT_OUTPUT = Path("data/processed/gb2760_a1_grouped_report.md")
ID_PREFIX = "GB2760_A1_TERM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group GB 2760-2024 A.1 rule JSONL into minimal additive-level JSONL."
    )
    parser.add_argument("--input", default="", help="Input A.1 rule JSONL path.")
    parser.add_argument(
        "--jsonl-output",
        default=str(DEFAULT_JSONL_OUTPUT),
        help="Output JSONL path for vector DB ingestion.",
    )
    parser.add_argument(
        "--pretty-output",
        default=str(DEFAULT_PRETTY_OUTPUT),
        help="Output pretty JSON path for human inspection.",
    )
    parser.add_argument(
        "--report-output",
        default=str(DEFAULT_REPORT_OUTPUT),
        help="Output Markdown report path.",
    )
    return parser.parse_args()


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def normalize_remark(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_code_sort_key(code: str) -> tuple[Any, ...]:
    parts = []
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
        lower_name = path.name.lower()
        score = 0
        if "gb2760" in lower_name:
            score += 10
        if "a1" in lower_name:
            score += 10
        if "rules" in lower_name:
            score += 8
        if "ready" in lower_name:
            score += 20
        if "processed" in str(path.parent).lower():
            score += 5
        if score > 0:
            ranked.append((score, path))

    if not ranked:
        raise FileNotFoundError("No suitable GB 2760 A.1 rules JSONL file was found.")

    ranked.sort(key=lambda item: (-item[0], len(str(item[1])), str(item[1])))
    return ranked[0][1]


def load_jsonl_records(input_path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    invalid_lines = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_lines += 1
                raise ValueError(f"Invalid JSON on line {line_no} of {input_path}: {exc}") from exc
            if not isinstance(data, dict):
                invalid_lines += 1
                raise ValueError(f"Expected a JSON object on line {line_no} of {input_path}.")
            records.append(data)
    return records, invalid_lines


def build_group_id(sample_record: dict[str, Any], normalized_term: str) -> str:
    seed = "|".join(
        [
            str(sample_record.get("standard_no", "GB 2760-2024")),
            str(sample_record.get("table_no", "A.1")),
            normalized_term,
        ]
    )
    short_hash = hashlib.md5(seed.encode("utf-8")).hexdigest()[:8]
    return f"{ID_PREFIX}_{short_hash}"


def choose_keyword_food_names(rules: list[dict[str, str]], limit: int = 5) -> list[str]:
    counter: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for index, rule in enumerate(rules):
        name = rule["food_category_name"]
        counter[name] += 1
        first_seen.setdefault(name, index)
    ordered = sorted(counter, key=lambda name: (-counter[name], first_seen[name], name))
    return ordered[:limit]


def summarize_embedding_text(term: str, function_category: str, rules: list[dict[str, str]]) -> str:
    food_names = dedupe_preserve_order([rule["food_category_name"] for rule in rules])[:10]
    usage_limits = dedupe_preserve_order([rule["usage_limit"] for rule in rules])[:6]
    remarks = dedupe_preserve_order([rule["remarks"] for rule in rules if rule["remarks"]])[:3]

    food_part = "、".join(food_names)
    usage_part = "、".join(usage_limits)
    text = (
        f"GB 2760-2024 A.1 规定，食品添加剂{term}属于{function_category}，"
        f"可用于{food_part}等食品。"
    )
    if usage_part:
        text += f"典型使用限量包括 {usage_part}。"
    if remarks:
        if len(remarks) == 1:
            text += f"备注多为“{remarks[0]}”。"
        else:
            text += f"部分条目标注备注：{'；'.join(remarks)}。"
    return text


def group_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    missing_normalized_term = 0
    missing_function_category = 0

    for record in records:
        normalized_term = str(record.get("normalized_term", "")).strip()
        function_category = str(record.get("function_category", "")).strip()
        if not normalized_term:
            missing_normalized_term += 1
            continue
        if not function_category:
            missing_function_category += 1
            continue
        groups[(normalized_term, function_category)].append(record)

    duplicate_rules_removed = 0
    empty_rule_groups = 0
    grouped_objects: list[dict[str, Any]] = []

    for (_, _), items in groups.items():
        items_sorted = sorted(
            items,
            key=lambda item: (
                parse_code_sort_key(str(item.get("food_category_code", ""))),
                str(item.get("food_category_name", "")),
                str(item.get("usage_limit", "")),
                str(item.get("remarks", "")),
            ),
        )
        unique_rules: list[dict[str, str]] = []
        seen_rule_keys: set[tuple[str, str, str, str]] = set()

        for item in items_sorted:
            rule = {
                "food_category_code": str(item.get("food_category_code", "")).strip(),
                "food_category_name": str(item.get("food_category_name", "")).strip(),
                "usage_limit": str(item.get("usage_limit", "")).strip(),
                "remarks": normalize_remark(item.get("remarks", "")),
            }
            key = (
                rule["food_category_code"],
                rule["food_category_name"],
                rule["usage_limit"],
                rule["remarks"],
            )
            if key in seen_rule_keys:
                duplicate_rules_removed += 1
                continue
            seen_rule_keys.add(key)
            unique_rules.append(rule)

        unique_rules.sort(key=lambda rule: parse_code_sort_key(rule["food_category_code"]))
        if not unique_rules:
            empty_rule_groups += 1
            continue

        sample = items_sorted[0]
        normalized_term = str(sample.get("normalized_term", "")).strip()
        function_category = str(sample.get("function_category", "")).strip()
        term = str(sample.get("term", normalized_term)).strip() or normalized_term
        keywords = dedupe_preserve_order(
            [normalized_term, function_category, "GB2760"] + choose_keyword_food_names(unique_rules)
        )
        grouped = {
            "id": build_group_id(sample, normalized_term),
            "term": term,
            "normalized_term": normalized_term,
            "function_category": function_category,
            "rules": unique_rules,
            "keywords": keywords,
            "embedding_text": summarize_embedding_text(term, function_category, unique_rules),
        }
        grouped_objects.append(grouped)

    grouped_objects.sort(key=lambda item: (item["term"], item["function_category"], item["id"]))

    stats = {
        "missing_normalized_term": missing_normalized_term,
        "missing_function_category": missing_function_category,
        "duplicate_rules_removed": duplicate_rules_removed,
        "empty_rule_groups": empty_rule_groups,
    }
    return grouped_objects, stats


def write_jsonl(output_path: Path, objects: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for obj in objects:
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_pretty_json(output_path: Path, objects: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(objects, handle, ensure_ascii=False, indent=2)


def build_report(
    input_path: Path,
    jsonl_output: Path,
    pretty_output: Path,
    report_output: Path,
    raw_count: int,
    grouped_count: int,
    duplicate_rules_removed: int,
    missing_normalized_term: int,
    missing_function_category: int,
    grouped_objects: list[dict[str, Any]],
) -> str:
    rng = random.Random(2760)
    samples = grouped_objects[:]
    if len(samples) > 3:
        samples = rng.sample(samples, 3)
    sample_lines = "\n".join(
        f"- `{item['term']}`: `{len(item['rules'])}` rules" for item in sorted(samples, key=lambda item: item['term'])
    )
    return "\n".join(
        [
            "# GB2760 A.1 Grouped Report",
            "",
            f"- 输入文件路径：`{input_path.resolve()}`",
            f"- JSONL 输出路径：`{jsonl_output.resolve()}`",
            f"- Pretty JSON 输出路径：`{pretty_output.resolve()}`",
            f"- 报告输出路径：`{report_output.resolve()}`",
            f"- 原始明细记录数：`{raw_count}`",
            f"- 聚合后记录数：`{grouped_count}`",
            f"- 去重掉的重复 rules 数量：`{duplicate_rules_removed}`",
            f"- 缺失 normalized_term 的记录数：`{missing_normalized_term}`",
            f"- 缺失 function_category 的记录数：`{missing_function_category}`",
            "",
            "## 随机抽样 3 个 term 的 rule_count",
            sample_lines or "- 无可用样本",
        ]
    ) + "\n"


def write_report(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def main() -> int:
    args = parse_args()
    try:
        input_path = resolve_input_path(args.input)
        jsonl_output = Path(args.jsonl_output)
        pretty_output = Path(args.pretty_output)
        report_output = Path(args.report_output)

        records, invalid_lines = load_jsonl_records(input_path)
        if invalid_lines:
            raise RuntimeError(f"Invalid JSON lines detected in {input_path}: {invalid_lines}")

        grouped_objects, stats = group_records(records)

        write_jsonl(jsonl_output, grouped_objects)
        write_pretty_json(pretty_output, grouped_objects)
        report = build_report(
            input_path=input_path,
            jsonl_output=jsonl_output,
            pretty_output=pretty_output,
            report_output=report_output,
            raw_count=len(records),
            grouped_count=len(grouped_objects),
            duplicate_rules_removed=stats["duplicate_rules_removed"],
            missing_normalized_term=stats["missing_normalized_term"],
            missing_function_category=stats["missing_function_category"],
            grouped_objects=grouped_objects,
        )
        write_report(report_output, report)

        summary = {
            "input": str(input_path.resolve()),
            "jsonl_output": str(jsonl_output.resolve()),
            "pretty_output": str(pretty_output.resolve()),
            "report_output": str(report_output.resolve()),
            "raw_count": len(records),
            "grouped_count": len(grouped_objects),
            "duplicate_rules_removed": stats["duplicate_rules_removed"],
            "missing_normalized_term": stats["missing_normalized_term"],
            "missing_function_category": stats["missing_function_category"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # pragma: no cover - top-level safety
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
