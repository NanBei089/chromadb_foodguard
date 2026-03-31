#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Retrieve all ingredient items against the GB2760 grouped Chroma collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import traceback
import unicodedata
from pathlib import Path
from typing import Any

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = PROJECT_ROOT / "ingredient_rag_pipeline_all"
DEFAULT_INPUT = Path(r"E:\GraduationProject\project\demo\upload_results\00f5f6bc-000003\other.ingredients.json")
DEFAULT_JSONL_CANDIDATES = [
    PROJECT_ROOT / "data/processed/gb2760_a1_grouped_min_final_prod.jsonl",
    PROJECT_ROOT / "data/processed/gb2760_a1_grouped_min_final_v2.jsonl",
]
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION = "gb2760_a1_grouped"
DEFAULT_OUTPUT = PIPELINE_ROOT / "data/ingredient_rag_results.json"
DEFAULT_REPORT = PIPELINE_ROOT / "reports/ingredient_rag_report.md"
VECTOR_DIM = 512


def log(message: str) -> None:
    print(f"[ingredient_rag] {message}")


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_text(value: Any) -> str:
    text = normalize_text(value).lower()
    replacements = {
        "（": "(",
        "）": ")",
        "，": ",",
        "、": ",",
        "；": ",",
        "：": ":",
        "’": "'",
        "′": "'",
        "＇": "'",
        "－": "-",
        "—": "-",
        "‐": "-",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"[()'\",;:\-·•\s]", "", text)
    return text


def feature_tokens(text: str) -> list[tuple[str, float]]:
    clean_text = normalize_text(text)
    if not clean_text:
        return []

    compact = clean_text.replace(" ", "")
    tokens: list[tuple[str, float]] = []
    for ch in compact:
        tokens.append((f"c:{ch}", 1.0))
    for size, weight in ((2, 1.9), (3, 2.5)):
        if len(compact) >= size:
            for index in range(len(compact) - size + 1):
                tokens.append((f"n{size}:{compact[index:index + size]}", weight))
    for word in re.findall(r"[\u4e00-\u9fffA-Za-z0-9.+≤≥/%-]+", clean_text):
        tokens.append((f"w:{word}", 2.2))
    return tokens


def embed_text(text: str, dim: int = VECTOR_DIM) -> list[float]:
    vector = [0.0] * dim
    for token, weight in feature_tokens(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * weight

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def resolve_input_path(cli_input: str | None) -> Path:
    if cli_input:
        path = Path(cli_input)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path
    if DEFAULT_INPUT.exists():
        return DEFAULT_INPUT
    raise FileNotFoundError(f"Input file not found: {DEFAULT_INPUT}")


def resolve_jsonl_path(cli_jsonl: str | None) -> Path:
    if cli_jsonl:
        path = Path(cli_jsonl)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")
        return path
    for candidate in DEFAULT_JSONL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No grouped JSONL file was found.")


def load_records_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            record_id = normalize_text(obj.get("id"))
            if record_id:
                rows[record_id] = obj
    return rows


def read_json_text_with_fallback(path: Path) -> str:
    encodings = ["utf-8", "utf-8-sig", "gb18030", "cp936"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    return path.read_text(encoding="utf-8")


def load_input_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(read_json_text_with_fallback(path))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Input JSON field 'items' must be a list.")
    return payload


def keyword_matches_query(query_term: str, candidate: str) -> bool:
    return canonical_text(query_term) == canonical_text(candidate)


def iter_candidate_strings(record: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("term", "normalized_term"):
        text = normalize_text(record.get(key))
        if text:
            values.append(text)

    aliases = record.get("aliases") if isinstance(record.get("aliases"), list) else []
    values.extend(normalize_text(item) for item in aliases if normalize_text(item))

    keywords = record.get("keywords") if isinstance(record.get("keywords"), list) else []
    values.extend(normalize_text(item) for item in keywords if normalize_text(item))
    return values


def record_matches_query(query_term: str, record: dict[str, Any]) -> bool:
    query_c = canonical_text(query_term)
    if not query_c:
        return False

    for key in ("term", "normalized_term"):
        candidate_c = canonical_text(record.get(key, ""))
        if candidate_c and query_c in candidate_c:
            return True

    aliases = record.get("aliases") if isinstance(record.get("aliases"), list) else []
    for alias in aliases:
        candidate_c = canonical_text(alias)
        if candidate_c and query_c in candidate_c:
            return True

    keywords = record.get("keywords") if isinstance(record.get("keywords"), list) else []
    for keyword in keywords:
        if keyword_matches_query(query_term, keyword):
            return True

    return False


def fetch_chroma_results(client: chromadb.PersistentClient, collection_name: str, query_text: str, n_results: int) -> list[dict[str, Any]]:
    collection = client.get_collection(name=collection_name)
    response = collection.query(
        query_embeddings=[embed_text(query_text)],
        n_results=n_results,
        include=["documents", "metadatas"],
    )
    ids = (response.get("ids") or [[]])[0]
    documents = (response.get("documents") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]

    results: list[dict[str, Any]] = []
    for index, record_id in enumerate(ids):
        document = documents[index] if index < len(documents) else ""
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        results.append(
            {
                "id": record_id,
                "document": normalize_text(document),
                "metadata": metadata,
                "rank": index + 1,
            }
        )
    return results


def backfill_matches(raw_results: list[dict[str, Any]], records_by_id: dict[str, dict[str, Any]], query_term: str) -> tuple[str, list[dict[str, Any]], bool]:
    matched_entries: list[dict[str, Any]] = []
    top1_is_match = False

    for item in raw_results:
        record_id = normalize_text(item.get("id"))
        source = records_by_id.get(record_id)
        if not source:
            continue
        if not record_matches_query(query_term, source):
            continue

        entry = {
            "id": source.get("id", ""),
            "term": source.get("term", ""),
            "normalized_term": source.get("normalized_term", ""),
            "aliases": source.get("aliases", []) if isinstance(source.get("aliases"), list) else [],
            "function_category": source.get("function_category", ""),
            "rules": source.get("rules", []) if isinstance(source.get("rules"), list) else [],
            "keywords": source.get("keywords", []) if isinstance(source.get("keywords"), list) else [],
            "score_rank": int(item.get("rank", 0)),
        }
        matched_entries.append(entry)
        if entry["score_rank"] == 1:
            top1_is_match = True

    if top1_is_match:
        quality = "high"
    elif matched_entries:
        quality = "weak"
    else:
        quality = "empty"

    return quality, matched_entries, bool(matched_entries)


def process_items(
    payload: dict[str, Any],
    source_file: str,
    records_by_id: dict[str, dict[str, Any]],
    chroma_dir: Path,
    collection_name: str,
    n_results: int,
) -> dict[str, Any]:
    client = chromadb.PersistentClient(path=str(chroma_dir))
    results: list[dict[str, Any]] = []

    for raw_item in payload.get("items", []):
        raw_term = ""
        if isinstance(raw_item, dict):
            raw_term = normalize_text(raw_item.get("term"))
        else:
            raw_term = normalize_text(raw_item)
        normalized_term = raw_term

        if not normalized_term:
            results.append(
                {
                    "raw_term": raw_term,
                    "normalized_term": normalized_term,
                    "retrieved": False,
                    "match_quality": "empty",
                    "matches": [],
                }
            )
            continue

        try:
            raw_matches = fetch_chroma_results(client, collection_name, normalized_term, n_results=n_results)
            quality, matches, retrieved = backfill_matches(raw_matches, records_by_id, normalized_term)
        except Exception as error:  # noqa: BLE001
            log(f"Query failed for term={normalized_term}: {error}")
            quality, matches, retrieved = "empty", [], False

        results.append(
            {
                "raw_term": raw_term,
                "normalized_term": normalized_term,
                "retrieved": retrieved,
                "match_quality": quality,
                "matches": matches if retrieved else [],
            }
        )

    return {
        "source_file": source_file,
        "ingredients_text": payload.get("ingredients_text", ""),
        "items_total": len(payload.get("items", [])),
        "retrieval_results": results,
    }


def build_report(
    input_path: Path,
    jsonl_path: Path,
    chroma_dir: Path,
    collection_name: str,
    result_payload: dict[str, Any],
) -> str:
    rows = result_payload["retrieval_results"]
    total = len(rows)
    high = [row for row in rows if row["match_quality"] == "high"]
    weak = [row for row in rows if row["match_quality"] == "weak"]
    empty = [row for row in rows if row["match_quality"] == "empty"]

    hit_examples = []
    for row in (high + weak)[:5]:
        top = row["matches"][0]["term"] if row["matches"] else ""
        hit_examples.append(f"- `{row['raw_term']}` -> `{top}` ({row['match_quality']})")

    empty_examples = [f"- `{row['raw_term']}`" for row in empty[:5]]

    recommend = "建议进入下一步“检索结果 + JSON 一起喂给模型”。" if (len(high) + len(weak)) >= 4 else "暂不建议直接进入下一步模型生成，建议先补强召回或做术语过滤。"

    lines = [
        "# Ingredient RAG Report",
        "",
        "## Overview",
        f"- 输入文件路径: `{input_path}`",
        f"- JSONL 路径: `{jsonl_path}`",
        f"- Chroma 路径: `{chroma_dir}`",
        f"- collection 名称: `{collection_name}`",
        "",
        "## Counts",
        f"- items 总数: `{total}`",
        f"- 实际检索术语数量: `{total}`",
        f"- 高质量命中数量: `{len(high)}`",
        f"- 弱命中数量: `{len(weak)}`",
        f"- 空结果数量: `{len(empty)}`",
        "",
        "## Hit Examples",
        *(hit_examples or ["- 无命中样例。"]),
        "",
        "## Empty Examples",
        *(empty_examples or ["- 无空结果样例。"]),
        "",
        "## Recommendation",
        f"- {recommend}",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve all ingredients against GB2760 Chroma.")
    parser.add_argument("--input", default=None, help="Path to other.ingredients.json")
    parser.add_argument("--jsonl", default=None, help="Path to grouped JSONL")
    parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR), help="Path to Chroma db directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSON output path")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Markdown report path")
    parser.add_argument("--n-results", type=int, default=3, help="Number of retrieval results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        input_path = resolve_input_path(args.input)
        jsonl_path = resolve_jsonl_path(args.jsonl)

        chroma_dir = Path(args.chroma_dir)
        if not chroma_dir.is_absolute():
            chroma_dir = PROJECT_ROOT / chroma_dir
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = PROJECT_ROOT / report_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        payload = load_input_payload(input_path)
        records_by_id = load_records_by_id(jsonl_path)
        result_payload = process_items(
            payload=payload,
            source_file=input_path.name,
            records_by_id=records_by_id,
            chroma_dir=chroma_dir,
            collection_name=args.collection,
            n_results=max(1, int(args.n_results)),
        )

        output_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path.write_text(
            build_report(
                input_path=input_path,
                jsonl_path=jsonl_path,
                chroma_dir=chroma_dir,
                collection_name=args.collection,
                result_payload=result_payload,
            ),
            encoding="utf-8",
        )

        rows = result_payload["retrieval_results"]
        high = sum(1 for row in rows if row["match_quality"] == "high")
        weak = sum(1 for row in rows if row["match_quality"] == "weak")
        empty = sum(1 for row in rows if row["match_quality"] == "empty")

        log(f"items 总数: {len(rows)}")
        log(f"高质量命中数量: {high}")
        log(f"弱命中数量: {weak}")
        log(f"空结果数量: {empty}")
        log(f"结果文件: {output_path}")
        log(f"报告文件: {report_path}")
        return 0
    except Exception as error:  # noqa: BLE001
        log(f"Fatal error: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
