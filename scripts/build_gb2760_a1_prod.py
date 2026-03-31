from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("data/processed/gb2760_a1_grouped_min_final_v2.jsonl")
DEFAULT_DEBUG = Path("data/processed/gb2760_a1_grouped_min_final_v2_debug.json")
DEFAULT_PROD = Path("data/processed/gb2760_a1_grouped_min_final_prod.jsonl")
DEFAULT_CONFLICT = Path("data/processed/gb2760_a1_conflict_records.json")
DEFAULT_REPORT = Path("data/processed/gb2760_a1_grouped_min_final_prod_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build production GB2760 A.1 JSONL by excluding conflict records.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input final v2 JSONL path.")
    parser.add_argument("--debug", default=str(DEFAULT_DEBUG), help="Input final v2 debug JSON path.")
    parser.add_argument("--prod-output", default=str(DEFAULT_PROD), help="Output production JSONL path.")
    parser.add_argument("--conflict-output", default=str(DEFAULT_CONFLICT), help="Output conflict JSON path.")
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT), help="Output report Markdown path.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()


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


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def conflict_key(term: str, normalized_term: str) -> tuple[str, str]:
    return normalize_text(term), normalize_text(normalized_term)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_report(path: Path, input_path: Path, prod_path: Path, conflict_path: Path, prod_count: int, conflict_count: int) -> None:
    content = "\n".join(
        [
            "# GB2760 A.1 Prod Report",
            "",
            f"- prod 记录数：`{prod_count}`",
            f"- conflict 记录数：`{conflict_count}`",
            f"- prod 文件路径：`{prod_path.resolve()}`",
            f"- conflict 文件路径：`{conflict_path.resolve()}`",
            f"- 输入文件路径：`{input_path.resolve()}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        input_path = Path(args.input)
        debug_path = Path(args.debug)
        prod_output = Path(args.prod_output)
        conflict_output = Path(args.conflict_output)
        report_output = Path(args.report_output)

        if not input_path.exists():
            raise FileNotFoundError(f"Input JSONL not found: {input_path}")
        if not debug_path.exists():
            raise FileNotFoundError(f"Debug JSON not found: {debug_path}")

        records = load_jsonl(input_path)
        debug_payload = load_json(debug_path)
        conflict_examples = debug_payload.get("conflict_examples") or []
        if not isinstance(conflict_examples, list):
            raise ValueError("debug conflict_examples must be a list")

        conflict_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item in conflict_examples:
            if not isinstance(item, dict):
                continue
            key = conflict_key(item.get("term", ""), item.get("normalized_term", ""))
            if not all(key):
                continue
            conflict_map[key] = item.get("conflicts") or []

        prod_rows: list[dict[str, Any]] = []
        conflict_rows: list[dict[str, Any]] = []
        matched_conflict_keys: set[tuple[str, str]] = set()

        for record in records:
            key = conflict_key(record.get("term", ""), record.get("normalized_term", ""))
            if key in conflict_map:
                matched_conflict_keys.add(key)
                conflict_rows.append(
                    {
                        "record": record,
                        "conflicts": conflict_map[key],
                    }
                )
            else:
                prod_rows.append(record)

        missing_matches = sorted(
            [
                {"term": key[0], "normalized_term": key[1]}
                for key in conflict_map.keys()
                if key not in matched_conflict_keys
            ],
            key=lambda item: (item["term"], item["normalized_term"]),
        )

        write_jsonl(prod_output, prod_rows)
        write_json(
            conflict_output,
            {
                "source_input": str(input_path.resolve()),
                "source_debug": str(debug_path.resolve()),
                "conflict_record_count": len(conflict_rows),
                "missing_conflict_matches": missing_matches,
                "records": conflict_rows,
            },
        )
        write_report(report_output, input_path, prod_output, conflict_output, len(prod_rows), len(conflict_rows))

        summary = {
            "prod_count": len(prod_rows),
            "conflict_count": len(conflict_rows),
            "prod_output": str(prod_output.resolve()),
            "conflict_output": str(conflict_output.resolve()),
            "report_output": str(report_output.resolve()),
            "missing_conflict_matches": len(missing_matches),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
