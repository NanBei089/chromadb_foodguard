from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import fitz
import pdfplumber

DOC_ID = "GB2760_2024_A1"
DOC_NAME = "GB 2760-2024 食品安全国家标准 食品添加剂使用标准"
STANDARD_NO = "GB 2760-2024"
SECTION = "附录A"
TABLE_NO = "A.1"
LANGUAGE = "zh-CN"
SOURCE_LEVEL = "official"
CHUNK_TYPE = "rule_entry"
TITLE_ONLY_MARKERS = {
    "表A.1（续）",
    "表 A.1（续）",
    "表A.1(续)",
    "表 A.1(续)",
    "表A.1 食品添加剂的使用范围和使用量",
}
TABLE_TITLE_RE = re.compile(r"^表\s*A\.1(?:[（(]续[)）])?(?:\s+食品添加剂的使用范围和使用量)?$")
NUMBER_ONLY_RE = re.compile(r"^\d+(?:\.\d+)?$")
POSSIBLE_NUMERIC_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)?|\.\d+)(?P<unit>[^\d\s].+)$")
PLAIN_NUMERIC_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)?|\.\d+)$")
INCOMPLETE_NUMERIC_RE = re.compile(r"^\d+\.$")
HEADER_HINT_RE = re.compile(r"食品分类号|食品名称|最大使用量|备注")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract GB 2760-2024 Appendix A Table A.1 rules.")
    parser.add_argument("--input", default="data/GB2760.pdf", help="Path to the GB 2760-2024 PDF.")
    parser.add_argument("--output-dir", default="output", help="Directory for output artifacts.")
    return parser.parse_args()


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = text.replace("\r", "\n").replace("\u3000", " ").replace("\xa0", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
        text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[，。；：、）】》〉％℃])", "", text)
        text = re.sub(r"(?<=[（【《〈])\s+(?=[\u4e00-\u9fff])", "", text)
        text = re.sub(r"(?<=\d)\s+(?=(?:mg|g|kg|L|mL|ml|IU|份|dm2))", "", text)
        text = re.sub(r"(?<=≤|≥|<|>)\s+", "", text)
        text = re.sub(r"(?<=[A-Za-z])\s+(?=号)", "", text)
    return text.strip()


def clean_region_lines(region_text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in region_text.splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue
        if line.startswith("GB2760-"):
            continue
        if re.fullmatch(r"\d+", line):
            continue
        lines.append(line)
    return lines


def parse_additive_metadata(region_text: str) -> Optional[dict[str, str]]:
    lines = clean_region_lines(region_text)
    informative = [line for line in lines if not TABLE_TITLE_RE.match(line)]
    if not informative:
        return None

    function_category = ""
    for idx, line in enumerate(informative):
        if line.startswith("功能"):
            function_category = normalize_text(line.removeprefix("功能"))
            if not function_category and idx + 1 < len(informative):
                function_category = informative[idx + 1]
            break

    term_lines: list[str] = []
    for line in informative:
        if line.startswith("CNS号") or line.startswith("INS号") or line.startswith("功能"):
            break
        term_lines.append(line)

    term_tokens = [token for token in re.split(r"\s+", " ".join(term_lines)) if token and has_cjk(token)]
    term = normalize_text("".join(term_tokens))
    term = re.sub(r"^[，。、；：]+|[，。、；：]+$", "", term)

    if not term or not function_category:
        return None

    return {
        "term": term,
        "function_category": function_category,
        "raw_region": " | ".join(informative),
    }


def parse_header_unit(header_cell: str) -> Optional[str]:
    header = normalize_text(header_cell)
    match = re.search(r"[（(]([^()（）]+)[)）]", header)
    if match:
        return normalize_text(match.group(1))
    return None


def append_fragment(base: str, fragment: str) -> str:
    if not fragment:
        return normalize_text(base)
    return normalize_text(f"{base}{fragment}")


def classify_limit(usage_raw: str, header_unit: Optional[str]) -> tuple[str, str, Optional[float], Optional[str], bool]:
    usage = normalize_text(usage_raw)
    compact = usage.replace(" ", "")
    if not usage:
        return "", "text", None, None, False
    if "按生产需要适量使用" in compact:
        return "按生产需要适量使用", "qs", None, None, False
    if "不得使用" in compact:
        return "不得使用", "forbidden", None, None, False
    if INCOMPLETE_NUMERIC_RE.fullmatch(compact):
        return usage, "text", None, None, True

    plain_match = PLAIN_NUMERIC_RE.fullmatch(compact)
    if plain_match and header_unit:
        value = float(plain_match.group("num"))
        unit = normalize_text(header_unit)
        return f"{plain_match.group('num')} {unit}", "numeric", value, unit, False

    explicit_match = POSSIBLE_NUMERIC_RE.fullmatch(compact)
    if explicit_match:
        value = float(explicit_match.group("num"))
        unit = normalize_text(explicit_match.group("unit"))
        return f"{explicit_match.group('num')} {unit}", "numeric", value, unit, False

    return usage, "text", None, None, False


def unique_keywords(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = normalize_text(value)
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def build_embedding_text(record: dict[str, Any]) -> str:
    sentence = (
        f"根据 {STANDARD_NO} {SECTION}表{TABLE_NO}，食品添加剂{record['term']}属于"
        f"{record['function_category']}，在{record['food_category_name']}中最大使用量为{record['usage_limit']}。"
    )
    if record["remarks"]:
        sentence += f"备注：{record['remarks']}。"
    return sentence


def safe_id_component(value: str, keep_dots: bool = False) -> str:
    pattern = r"[^\w\u4e00-\u9fff.-]+" if keep_dots else r"[^\w\u4e00-\u9fff-]+"
    value = normalize_text(value)
    value = re.sub(pattern, "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def build_record_id(term: str, food_category_code: str, seen_ids: dict[str, int], debug_lines: list[str]) -> str:
    prefix = "GB2760_2024_A1"
    term_part = safe_id_component(term)
    code_part = safe_id_component(food_category_code, keep_dots=True)
    base_id = f"{prefix}_{term_part}_{code_part}"
    index = seen_ids.get(base_id, 0) + 1
    seen_ids[base_id] = index
    if index == 1:
        return base_id
    resolved = f"{base_id}__{index}"
    debug_lines.append(
        f"[ID_COLLISION] base_id={base_id} duplicate_index={index} term={term} food_category_code={food_category_code}"
    )
    return resolved


def refresh_record(record: dict[str, Any]) -> None:
    usage_limit, limit_type, limit_value, unit, suspicious = classify_limit(
        record["_usage_raw"], record.get("_header_unit")
    )
    record["usage_limit"] = usage_limit
    record["limit_type"] = limit_type
    record["limit_value"] = limit_value
    record["unit"] = unit
    record["_suspicious_usage"] = suspicious
    record["food_category_name"] = normalize_text(record["food_category_name"])
    record["remarks"] = normalize_text(record["remarks"])
    record["keywords"] = unique_keywords(
        [record["term"], record["function_category"], record["food_category_name"], "GB2760"]
    )
    record["embedding_text"] = build_embedding_text(record)


def clean_row(row: list[Optional[str]]) -> list[str]:
    return [normalize_text(cell) for cell in row]


def is_header_row(cells: list[str]) -> bool:
    joined = " ".join(cells)
    return "食品分类号" in joined and "食品名称" in joined and "最大使用量" in joined


def is_blank_row(cells: list[str]) -> bool:
    return not any(cells)


def locate_a1_pages(pdf_path: Path) -> tuple[int, int, int]:
    doc = fitz.open(pdf_path)
    start_page = 9
    end_page = 103
    next_a2_page = 104
    page_104_text = doc[next_a2_page - 1].get_text()
    doc.close()
    if "A.2" not in page_104_text:
        raise RuntimeError("Expected Table A.2 to start on PDF page 104, but the verification failed.")
    return start_page, end_page, next_a2_page


def finalize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rules_path = output_dir / "gb2760_a1_rules.jsonl"
    debug_path = output_dir / "gb2760_a1_debug.txt"
    preview_path = output_dir / "gb2760_a1_preview.json"

    start_page, end_page, next_a2_page = locate_a1_pages(pdf_path)

    debug_lines: list[str] = []
    debug_lines.append(f"[INFO] input_pdf={pdf_path.resolve()}")
    debug_lines.append(f"[INFO] a1_page_range={start_page}-{end_page}")
    debug_lines.append(f"[INFO] next_a2_page={next_a2_page}")

    records: list[dict[str, Any]] = []
    current_additive: Optional[dict[str, str]] = None
    current_header_unit: Optional[str] = None
    pending_meta_text = ""
    seen_ids: dict[str, int] = {}

    table_count = 0
    header_rows_skipped = 0
    blank_rows_skipped = 0
    continuation_rows_merged = 0
    missing_key_rows = 0
    metadata_parse_failures = 0
    tables_continued_from_previous = 0
    metadata_joined_from_previous_page = 0
    suspicious_usage_count = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_no in range(start_page, end_page + 1):
            page = pdf.pages[page_no - 1]
            tables = page.find_tables()
            debug_lines.append(f"[PAGE] page={page_no} tables={len(tables)}")
            table_count += len(tables)
            previous_bottom = 60.0

            for table_index, table in enumerate(tables):
                region_text = (page.crop((0, previous_bottom, page.width, table.bbox[1])).extract_text() or "").strip()
                previous_bottom = table.bbox[3]
                combined_meta_source = "\n".join(part for part in [pending_meta_text, region_text] if part.strip())
                metadata = parse_additive_metadata(combined_meta_source)
                if metadata is not None:
                    if pending_meta_text.strip() and metadata.get("term"):
                        metadata_joined_from_previous_page += 1
                    current_additive = {
                        "term": metadata["term"],
                        "function_category": metadata["function_category"],
                    }
                    pending_meta_text = ""
                else:
                    stripped_region = " | ".join(clean_region_lines(region_text))
                    if stripped_region and stripped_region not in TITLE_ONLY_MARKERS:
                        metadata_parse_failures += 1
                        debug_lines.append(
                            f"[META_UNRESOLVED] page={page_no} table={table_index} region={stripped_region}"
                        )
                    else:
                        tables_continued_from_previous += 1
                    pending_meta_text = ""

                table_rows = [clean_row(row) for row in table.extract()]
                has_header = bool(table_rows) and is_header_row(table_rows[0])
                if metadata is not None:
                    current_header_unit = parse_header_unit(table_rows[0][2]) if has_header else None
                elif has_header:
                    header_unit = parse_header_unit(table_rows[0][2])
                    if header_unit:
                        current_header_unit = header_unit
                if not current_additive:
                    debug_lines.append(
                        f"[TABLE_SKIPPED_NO_CONTEXT] page={page_no} table={table_index} reason=no_additive_context"
                    )
                    missing_key_rows += max(0, len(table_rows) - (1 if has_header else 0))
                    continue

                for row_index, cells in enumerate(table_rows):
                    if is_blank_row(cells):
                        blank_rows_skipped += 1
                        continue
                    if is_header_row(cells):
                        header_rows_skipped += 1
                        continue

                    code, food_name, usage_cell, remarks = cells
                    if not code:
                        if records:
                            target = records[-1]
                            before_name = target["food_category_name"]
                            before_usage = target["_usage_raw"]
                            before_remarks = target["remarks"]
                            if food_name:
                                target["food_category_name"] = append_fragment(target["food_category_name"], food_name)
                            if usage_cell:
                                target["_usage_raw"] = append_fragment(target["_usage_raw"], usage_cell)
                            if remarks:
                                target["remarks"] = append_fragment(target["remarks"], remarks)
                            refresh_record(target)
                            continuation_rows_merged += 1
                            debug_lines.append(
                                "[ROW_CONTINUATION] "
                                f"page={page_no} table={table_index} row={row_index} "
                                f"name_fragment={food_name!r} usage_fragment={usage_cell!r} remarks_fragment={remarks!r} "
                                f"target_id={target['id']}"
                            )
                            if target["_suspicious_usage"]:
                                suspicious_usage_count += 1
                                debug_lines.append(
                                    f"[SUSPICIOUS_USAGE] page={page_no} table={table_index} row={row_index} id={target['id']} usage={target['_usage_raw']}"
                                )
                        else:
                            missing_key_rows += 1
                            debug_lines.append(
                                f"[ROW_SKIPPED] page={page_no} table={table_index} row={row_index} reason=no_previous_record_for_continuation cells={cells}"
                            )
                        continue

                    if not food_name or not usage_cell:
                        missing_key_rows += 1
                        debug_lines.append(
                            f"[ROW_SKIPPED] page={page_no} table={table_index} row={row_index} reason=missing_key_field cells={cells}"
                        )
                        continue

                    record = {
                        "id": build_record_id(current_additive["term"], code, seen_ids, debug_lines),
                        "doc_id": DOC_ID,
                        "doc_name": DOC_NAME,
                        "standard_no": STANDARD_NO,
                        "section": SECTION,
                        "table_no": TABLE_NO,
                        "term": current_additive["term"],
                        "normalized_term": current_additive["term"],
                        "aliases": [],
                        "term_type": "食品添加剂",
                        "function_category": current_additive["function_category"],
                        "food_category_code": code,
                        "food_category_name": food_name,
                        "usage_limit": "",
                        "limit_type": "text",
                        "limit_value": None,
                        "unit": None,
                        "remarks": remarks,
                        "source_level": SOURCE_LEVEL,
                        "language": LANGUAGE,
                        "chunk_type": CHUNK_TYPE,
                        "keywords": [],
                        "embedding_text": "",
                        "_usage_raw": usage_cell,
                        "_header_unit": current_header_unit,
                        "_suspicious_usage": False,
                    }
                    refresh_record(record)
                    if record["_suspicious_usage"]:
                        suspicious_usage_count += 1
                        debug_lines.append(
                            f"[SUSPICIOUS_USAGE] page={page_no} table={table_index} row={row_index} id={record['id']} usage={record['_usage_raw']}"
                        )
                    records.append(record)

            tail_text = (page.crop((0, previous_bottom, page.width, page.height - 20)).extract_text() or "").strip()
            pending_lines = [line for line in clean_region_lines(tail_text) if not HEADER_HINT_RE.search(line)]
            pending_meta_text = "\n".join(pending_lines).strip()
            if pending_meta_text:
                debug_lines.append(f"[PAGE_TAIL] page={page_no} tail={pending_meta_text}")

    final_records = [finalize_record(record) for record in records]
    with rules_path.open("w", encoding="utf-8") as handle:
        for record in final_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with preview_path.open("w", encoding="utf-8") as handle:
        json.dump(final_records[:20], handle, ensure_ascii=False, indent=2)

    debug_lines.insert(1, f"[INFO] detected_tables={table_count}")
    debug_lines.insert(2, f"[INFO] extracted_rules={len(final_records)}")
    debug_lines.insert(3, f"[INFO] skipped_header_rows={header_rows_skipped}")
    debug_lines.insert(4, f"[INFO] skipped_blank_rows={blank_rows_skipped}")
    debug_lines.insert(5, f"[INFO] merged_continuation_rows={continuation_rows_merged}")
    debug_lines.insert(6, f"[INFO] skipped_missing_key_rows={missing_key_rows}")
    debug_lines.insert(7, f"[INFO] metadata_parse_failures={metadata_parse_failures}")
    debug_lines.insert(8, f"[INFO] continuation_tables={tables_continued_from_previous}")
    debug_lines.insert(9, f"[INFO] metadata_joined_from_previous_page={metadata_joined_from_previous_page}")
    debug_lines.insert(10, f"[INFO] suspicious_usage_entries={suspicious_usage_count}")

    with debug_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(debug_lines) + "\n")

    summary = {
        "a1_page_range": [start_page, end_page],
        "next_a2_page": next_a2_page,
        "detected_tables": table_count,
        "extracted_rules": len(final_records),
        "skipped_missing_key_rows": missing_key_rows,
        "metadata_parse_failures": metadata_parse_failures,
        "suspicious_usage_entries": suspicious_usage_count,
        "rules_path": str(rules_path.resolve()),
        "debug_path": str(debug_path.resolve()),
        "preview_path": str(preview_path.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
