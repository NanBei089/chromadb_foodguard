#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Retrieve ingredient candidates from Chroma and rerank them with rule-based filtering."""

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
PIPELINE_ROOT = PROJECT_ROOT / "ingredient_rag_pipeline_ranked"
DEFAULT_INPUT_FALLBACK = Path(
    r"E:\GraduationProject\project\demo\upload_results\9d26ee9e-IMG_20251031_173657\other.ingredients.json"
)
DEFAULT_JSONL_CANDIDATES = [
    PROJECT_ROOT / "data/processed/gb2760_a1_grouped_min_final_prod.jsonl",
    PROJECT_ROOT / "data/processed/gb2760_a1_grouped_min_final_v2.jsonl",
]
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION = "gb2760_a1_grouped"
DEFAULT_OUTPUT = PIPELINE_ROOT / "data/ingredient_rag_ranked_results.json"
DEFAULT_REPORT = PIPELINE_ROOT / "reports/ingredient_rag_ranked_report.md"
DEFAULT_N_RESULTS = 5
VECTOR_DIM = 512


def log(message: str) -> None:
    print(f"[ingredient_rag_ranked] {message}")


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_text(value: Any) -> str:
    text = normalize_text(value).lower()
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("，", ",").replace("、", ",").replace("；", ",")
    text = text.replace("：", ":").replace("－", "-").replace("—", "-").replace("‐", "-")
    text = re.sub(r"[()'\",;:\-\s·•]", "", text)
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


def resolve_input_path(cli_input: str | None) -> Path:
    if cli_input:
        path = Path(cli_input)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    if DEFAULT_INPUT_FALLBACK.exists():
        return DEFAULT_INPUT_FALLBACK

    search_root = PROJECT_ROOT.parent / "demo" / "upload_results"
    matches = sorted(search_root.glob("*/other.ingredients.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]

    raise FileNotFoundError("No other.ingredients.json file was found.")


def resolve_jsonl_path(cli_jsonl: str | None) -> Path:
    if cli_jsonl:
        path = Path(cli_jsonl)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"JSONL source file not found: {path}")
        return path
    for candidate in DEFAULT_JSONL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No grouped JSONL source file was found.")


def load_input_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(read_json_text_with_fallback(path))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Input JSON field 'items' must be a list.")
    return payload


def load_records_by_id(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            record_id = normalize_text(obj.get("id"))
            if record_id:
                records[record_id] = obj
    return records


def fetch_chroma_results(
    client: chromadb.PersistentClient,
    collection_name: str,
    query_text: str,
    n_results: int,
) -> list[dict[str, Any]]:
    collection = client.get_collection(name=collection_name)
    response = collection.query(
        query_embeddings=[embed_text(query_text)],
        n_results=n_results,
        include=["documents", "metadatas"],
    )
    ids = (response.get("ids") or [[]])[0]
    documents = (response.get("documents") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]

    rows: list[dict[str, Any]] = []
    for index, record_id in enumerate(ids):
        rows.append(
            {
                "id": normalize_text(record_id),
                "document": normalize_text(documents[index] if index < len(documents) else ""),
                "metadata": metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {},
                "score_rank": index + 1,
            }
        )
    return rows


def dedupe_list(values: list[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = normalize_text(item)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def split_text_parts(text: str) -> list[str]:
    return [normalize_text(part) for part in re.split(r"[、,，]", normalize_text(text)) if normalize_text(part)]


def extract_term_variants(text: str) -> list[str]:
    clean = normalize_text(text)
    if not clean:
        return []

    variants = [clean]
    parts = split_text_parts(clean)
    if len(parts) > 1:
        variants.extend(parts)

    if "及其" in clean:
        prefix, suffix = clean.split("及其", 1)
        prefix = normalize_text(prefix)
        suffix_parts = split_text_parts(suffix)
        for suffix_part in suffix_parts:
            if not prefix or not suffix_part:
                continue
            variants.append(prefix + suffix_part)
            if suffix_part.endswith("盐") and len(suffix_part) > 1:
                variants.append(prefix + suffix_part[:-1])

    return dedupe_list(variants)


def char_overlap_ratio(query_c: str, candidate_c: str) -> float:
    if not query_c or not candidate_c:
        return 0.0
    query_chars = set(query_c)
    candidate_chars = set(candidate_c)
    if not query_chars or not candidate_chars:
        return 0.0
    return len(query_chars & candidate_chars) / max(1, len(query_chars))


def list_contains_exact(query_c: str, values: list[str]) -> bool:
    return any(canonical_text(item) == query_c for item in values if normalize_text(item))


def list_contains_containment(query_c: str, values: list[str], minimum_length: int = 3) -> bool:
    if len(query_c) < minimum_length:
        return False
    return any(query_c in canonical_text(item) for item in values if normalize_text(item))


def extract_term_keywords(source: dict[str, Any], term: str, normalized_term: str, aliases: list[str], keywords: list[str], function_category: str) -> list[str]:
    food_names = {
        normalize_text(rule.get("food_category_name"))
        for rule in source.get("rules", [])
        if isinstance(rule, dict) and normalize_text(rule.get("food_category_name"))
    }
    excluded = {normalize_text(function_category), "GB2760", "GB 2760", "GB2760-2024"}
    excluded.update(food_names)

    term_keywords: list[str] = []
    term_keywords.extend(extract_term_variants(term))
    term_keywords.extend(extract_term_variants(normalized_term))
    term_keywords.extend(extract_term_variants(alias) for alias in aliases)

    flattened_alias_variants: list[str] = []
    for alias in aliases:
        flattened_alias_variants.extend(extract_term_variants(alias))
    term_keywords.extend(flattened_alias_variants)

    for keyword in keywords:
        keyword_text = normalize_text(keyword)
        if not keyword_text or keyword_text in excluded:
            continue
        term_keywords.extend(extract_term_variants(keyword_text))

    return dedupe_list(term_keywords)


def build_match_entry(source: dict[str, Any], score_rank: int) -> dict[str, Any]:
    term = normalize_text(source.get("term"))
    normalized_term = normalize_text(source.get("normalized_term"))
    aliases = dedupe_list(source.get("aliases") if isinstance(source.get("aliases"), list) else [])
    keywords = dedupe_list(source.get("keywords") if isinstance(source.get("keywords"), list) else [])
    function_category = normalize_text(source.get("function_category"))
    return {
        "id": normalize_text(source.get("id")),
        "term": term,
        "normalized_term": normalized_term,
        "aliases": aliases,
        "function_category": function_category,
        "rules": source.get("rules") if isinstance(source.get("rules"), list) else [],
        "keywords": keywords,
        "match_keywords": extract_term_keywords(source, term, normalized_term, aliases, keywords, function_category),
        "embedding_text": normalize_text(source.get("embedding_text")),
        "score_rank": score_rank,
    }


def evaluate_candidate(query_term: str, candidate: dict[str, Any]) -> dict[str, Any]:
    query_c = canonical_text(query_term)
    term = candidate["term"]
    normalized_term = candidate["normalized_term"]
    aliases = candidate["aliases"]
    match_keywords = candidate["match_keywords"]

    term_c = canonical_text(term)
    normalized_c = canonical_text(normalized_term)
    keyword_exact = list_contains_exact(query_c, match_keywords)
    keyword_contains = list_contains_containment(query_c, match_keywords, minimum_length=3)
    alias_exact = list_contains_exact(query_c, aliases)
    alias_contains = list_contains_containment(query_c, aliases, minimum_length=3)
    exact_normalized = bool(query_c and query_c == normalized_c)
    exact_term = bool(query_c and query_c == term_c)
    contains_normalized = bool(query_c and len(query_c) >= 3 and normalized_c and query_c in normalized_c and query_c != normalized_c)
    contains_term = bool(query_c and len(query_c) >= 3 and term_c and query_c in term_c and query_c != term_c)

    overlap_targets = [term, normalized_term, *match_keywords]
    strong_overlap = False
    if len(query_c) >= 3:
        strong_overlap = any(char_overlap_ratio(query_c, canonical_text(value)) >= 0.85 for value in overlap_targets if normalize_text(value))

    score = 0
    if exact_normalized:
        score += 100
    elif contains_normalized:
        score += 80

    if exact_term:
        score += 95
    elif contains_term:
        score += 70

    if alias_exact:
        score += 90
    elif alias_contains:
        score += 55

    if keyword_exact:
        score += 60
    elif keyword_contains:
        score += 35

    if strong_overlap:
        score += 25

    score += {1: 30, 2: 20, 3: 10}.get(candidate["score_rank"], 5)

    direct_hit = exact_normalized or exact_term or contains_normalized or contains_term or alias_exact or keyword_exact
    weak_hit = not direct_hit and len(query_c) >= 3 and (alias_contains or keyword_contains or strong_overlap)

    if not direct_hit and len(term_c) >= max(8, len(query_c) * 2):
        score -= 20

    if not direct_hit and not weak_hit:
        score -= 30

    candidate["match_score"] = score
    candidate["is_high_match"] = direct_hit
    candidate["is_weak_match"] = weak_hit and score >= 45
    candidate["is_primary"] = False
    return candidate


def rerank_candidates(
    query_term: str,
    raw_results: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
) -> tuple[str, str, list[dict[str, Any]], int, list[str]]:
    scored_candidates: list[dict[str, Any]] = []
    filtered_examples: list[str] = []

    for raw in raw_results:
        source = records_by_id.get(raw["id"])
        if not source:
            continue
        candidate = build_match_entry(source, raw["score_rank"])
        candidate = evaluate_candidate(query_term, candidate)
        scored_candidates.append(candidate)

    scored_candidates.sort(key=lambda item: (-item["match_score"], item["score_rank"], item["term"]))
    high_candidates = [item for item in scored_candidates if item["is_high_match"]]
    weak_candidates = [item for item in scored_candidates if item["is_weak_match"]]

    if high_candidates:
        selected = [high_candidates[0]]
        if len(high_candidates) > 1 and high_candidates[0]["match_score"] - high_candidates[1]["match_score"] <= 20:
            selected.append(high_candidates[1])
        quality = "high"
    elif weak_candidates:
        selected = [weak_candidates[0]]
        quality = "weak"
    else:
        selected = []
        quality = "empty"

    selected_ids = {item["id"] for item in selected}
    filtered_count = 0
    for item in scored_candidates:
        if item["id"] not in selected_ids:
            filtered_count += 1
            if len(filtered_examples) < 3:
                filtered_examples.append(f"{query_term} -> {item['term']} ({item['match_score']})")

    if selected:
        selected[0]["is_primary"] = True
        primary_match_id = selected[0]["id"]
    else:
        primary_match_id = ""

    for item in selected:
        item.pop("embedding_text", None)
        item.pop("match_keywords", None)
        item.pop("is_high_match", None)
        item.pop("is_weak_match", None)

    return quality, primary_match_id, selected[:2], filtered_count, filtered_examples


def build_result_entry(
    query_term: str,
    raw_results: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], int, list[str]]:
    quality, primary_match_id, matches, filtered_count, filtered_examples = rerank_candidates(
        query_term=query_term,
        raw_results=raw_results,
        records_by_id=records_by_id,
    )
    retrieved = bool(matches)
    return (
        {
            "raw_term": query_term,
            "normalized_term": query_term,
            "retrieved": retrieved,
            "match_quality": quality,
            "primary_match_id": primary_match_id if retrieved else "",
            "matches": matches if retrieved else [],
        },
        filtered_count,
        filtered_examples,
    )


def process_payload(
    payload: dict[str, Any],
    source_file: Path,
    records_by_id: dict[str, dict[str, Any]],
    chroma_dir: Path,
    collection_name: str,
    n_results: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    client = chromadb.PersistentClient(path=str(chroma_dir))
    retrieval_results: list[dict[str, Any]] = []
    filtered_interference_count = 0
    examples_hit: list[str] = []
    examples_empty: list[str] = []
    examples_interference: list[str] = []
    retrieved_count = 0
    high_count = 0
    weak_count = 0
    empty_count = 0
    primary_count = 0

    for raw_item in payload.get("items", []):
        raw_term = normalize_text(raw_item.get("term") if isinstance(raw_item, dict) else raw_item)
        normalized_term = normalize_text(raw_term)
        if not normalized_term:
            retrieval_results.append(
                {
                    "raw_term": raw_term,
                    "normalized_term": normalized_term,
                    "retrieved": False,
                    "match_quality": "empty",
                    "primary_match_id": "",
                    "matches": [],
                }
            )
            empty_count += 1
            continue

        try:
            raw_results = fetch_chroma_results(
                client=client,
                collection_name=collection_name,
                query_text=normalized_term,
                n_results=n_results,
            )
        except Exception as error:  # noqa: BLE001
            log(f"Query failed for term={normalized_term}: {error}")
            raw_results = []

        result, filtered_count, filtered_examples = build_result_entry(normalized_term, raw_results, records_by_id)
        retrieval_results.append(result)
        filtered_interference_count += filtered_count

        if result["retrieved"]:
            retrieved_count += 1
            primary_count += 1
            match = result["matches"][0]
            if result["match_quality"] == "high":
                high_count += 1
            else:
                weak_count += 1
            if len(examples_hit) < 5:
                examples_hit.append(f"{normalized_term} -> {match['term']} ({result['match_quality']}, {match['match_score']})")
        else:
            empty_count += 1
            if len(examples_empty) < 5:
                examples_empty.append(normalized_term)

        for example in filtered_examples:
            if len(examples_interference) >= 5:
                break
            examples_interference.append(example)

    output = {
        "source_file": source_file.name,
        "ingredients_text": payload.get("ingredients_text", ""),
        "items_total": len(payload.get("items", [])),
        "retrieval_results": retrieval_results,
    }
    stats = {
        "input_file": str(source_file.resolve()),
        "items_total": len(payload.get("items", [])),
        "retrieved_total": retrieved_count,
        "high_quality_count": high_count,
        "weak_quality_count": weak_count,
        "empty_count": empty_count,
        "primary_kept_count": primary_count,
        "filtered_interference_count": filtered_interference_count,
        "hit_examples": examples_hit,
        "empty_examples": examples_empty,
        "interference_examples": examples_interference,
    }
    return output, stats


def build_report(
    stats: dict[str, Any],
    jsonl_source: Path,
    chroma_dir: Path,
    collection_name: str,
    result_path: Path,
) -> str:
    recommended = "yes" if stats["high_quality_count"] >= max(1, stats["items_total"] // 5) else "no"
    lines = [
        "# Ingredient RAG Ranked Report",
        "",
        "## Overview",
        f"- Input file: `{stats['input_file']}`",
        f"- JSONL source: `{jsonl_source.resolve()}`",
        f"- Chroma dir: `{chroma_dir.resolve()}`",
        f"- Collection: `{collection_name}`",
        f"- Result file: `{result_path.resolve()}`",
        "",
        "## Counts",
        f"- Items total: `{stats['items_total']}`",
        f"- Retrieved total: `{stats['retrieved_total']}`",
        f"- High quality count: `{stats['high_quality_count']}`",
        f"- Weak quality count: `{stats['weak_quality_count']}`",
        f"- Empty count: `{stats['empty_count']}`",
        f"- Primary kept count: `{stats['primary_kept_count']}`",
        f"- Filtered interference count: `{stats['filtered_interference_count']}`",
        "",
        "## Hit Examples",
    ]
    if stats["hit_examples"]:
        lines.extend(f"- {item}" for item in stats["hit_examples"])
    else:
        lines.append("- None")

    lines.extend(["", "## Empty Examples"])
    if stats["empty_examples"]:
        lines.extend(f"- {item}" for item in stats["empty_examples"])
    else:
        lines.append("- None")

    lines.extend(["", "## Interference Examples"])
    if stats["interference_examples"]:
        lines.extend(f"- {item}" for item in stats["interference_examples"])
    else:
        lines.append("- None")

    lines.extend(["", "## Recommendation", f"- Suggest next-step model generation: `{recommended}`"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve ingredient candidates and rerank them after Chroma recall.")
    parser.add_argument("--input", default=None, help="Path to other.ingredients.json")
    parser.add_argument("--jsonl-source", default=None, help="Path to grouped JSONL source")
    parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR), help="Path to Chroma persistent directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Ranked JSON output path")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Markdown report output path")
    parser.add_argument("--n-results", type=int, default=DEFAULT_N_RESULTS, help="Number of raw Chroma candidates to recall")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        input_path = resolve_input_path(args.input)
        jsonl_path = resolve_jsonl_path(args.jsonl_source)
        chroma_dir = Path(args.chroma_dir)
        if not chroma_dir.is_absolute():
            chroma_dir = PROJECT_ROOT / chroma_dir

        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = PROJECT_ROOT / report_path

        payload = load_input_payload(input_path)
        records_by_id = load_records_by_id(jsonl_path)
        output, stats = process_payload(
            payload=payload,
            source_file=input_path,
            records_by_id=records_by_id,
            chroma_dir=chroma_dir,
            collection_name=args.collection,
            n_results=max(1, int(args.n_results)),
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

        report_text = build_report(stats, jsonl_path, chroma_dir, args.collection, output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")

        log(f"items total: {stats['items_total']}")
        log(f"high quality count: {stats['high_quality_count']}")
        log(f"weak quality count: {stats['weak_quality_count']}")
        log(f"empty count: {stats['empty_count']}")
        log(f"filtered interference count: {stats['filtered_interference_count']}")
        log(f"result file: {output_path.resolve()}")
        log(f"report file: {report_path.resolve()}")
        return 0
    except Exception as error:  # noqa: BLE001
        log(f"fatal error: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

