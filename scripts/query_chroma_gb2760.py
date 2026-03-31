#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run validation queries against the local GB 2760 Chroma collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import traceback
import unicodedata
from pathlib import Path
from typing import Any

import chromadb

DEFAULT_DB_DIR = Path("chroma_db")
DEFAULT_COLLECTION = "gb2760_a1_grouped"
DEFAULT_QUERIES = [
    "安赛蜜是什么",
    "乙酰磺胺酸钾可以用于哪些食品",
    "亚硝酸钠能用于什么",
    "二氧化硫残留量",
    "卡拉胶是什么类别",
]
VECTOR_DIM = 512
STOP_TOKENS = {"是什么", "什么", "哪些", "可以", "用于", "能用于", "类别", "食品", "可以用于"}


def log(message: str) -> None:
    print(f"[query] {message}")


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_text(value: Any) -> str:
    return normalize_text(value).replace(" ", "")


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


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def summarize_document(text: str, limit: int = 120) -> str:
    clean = normalize_text(text)
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


def get_collection(client: chromadb.PersistentClient, collection_name: str):
    try:
        return client.get_collection(name=collection_name)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"Collection not found or cannot be opened: {collection_name}. {error}") from error


def split_candidates(*values: Any) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_text(value)
        if not text:
            continue
        parts = re.split(r"[、,，;；/（）()]+", text)
        for part in [text, *parts]:
            candidate = normalize_text(part)
            if len(candidate) >= 2 and candidate not in seen:
                seen.add(candidate)
                output.append(candidate)
    return output


def build_search_text(metadata: dict[str, Any], document: str) -> str:
    pieces: list[str] = []
    for field in ("term", "normalized_term", "aliases_text", "function_category"):
        value = metadata.get(field)
        text = normalize_text(value)
        if text:
            pieces.extend([text, text])
    keywords = metadata.get("keywords")
    if isinstance(keywords, list):
        for keyword in keywords:
            keyword_text = normalize_text(keyword)
            if keyword_text:
                pieces.append(keyword_text)
    pieces.append(normalize_text(document))
    return " | ".join(piece for piece in pieces if piece)


def query_tokens(query: str) -> list[str]:
    raw_tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9.+≤≥/%-]+", normalize_text(query))
    tokens = []
    seen = set()
    for token in raw_tokens:
        compact = compact_text(token)
        if len(compact) >= 2 and compact not in STOP_TOKENS and compact not in seen:
            seen.add(compact)
            tokens.append(compact)
    return tokens


def lexical_boost(query: str, metadata: dict[str, Any], document: str) -> float:
    compact_query = compact_text(query)
    if not compact_query:
        return 0.0

    boost = 0.0
    term = normalize_text(metadata.get("term"))
    normalized_term = normalize_text(metadata.get("normalized_term"))
    aliases = split_candidates(metadata.get("aliases_text"))
    keywords = metadata.get("keywords") if isinstance(metadata.get("keywords"), list) else []
    core_candidates = split_candidates(term, normalized_term, *aliases)
    keyword_candidates = [candidate for candidate in split_candidates(*keywords) if len(compact_text(candidate)) >= 4]
    document_text = compact_text(document)

    matched_core = False
    for candidate in core_candidates:
        compact_candidate = compact_text(candidate)
        if compact_candidate == compact_query:
            boost += 24.0
            matched_core = True
        elif compact_candidate in compact_query or compact_query in compact_candidate:
            boost += 18.0
            matched_core = True

    for candidate in keyword_candidates:
        compact_candidate = compact_text(candidate)
        if compact_candidate and (compact_candidate in compact_query or compact_query in compact_candidate):
            boost += 3.0

    for token in query_tokens(query):
        if token in document_text:
            boost += 0.8

    if matched_core and "类别" in query and normalize_text(metadata.get("function_category")):
        boost += 1.0
    return boost


def load_collection_rows(collection: Any) -> list[dict[str, Any]]:
    total = collection.count()
    batch = collection.get(limit=total, include=["documents", "metadatas"])
    ids = batch.get("ids") or []
    documents = batch.get("documents") or []
    metadatas = batch.get("metadatas") or []

    rows: list[dict[str, Any]] = []
    for index, record_id in enumerate(ids):
        document = documents[index] if index < len(documents) else ""
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        search_text = build_search_text(metadata, document)
        rows.append(
            {
                "id": record_id,
                "document": document,
                "metadata": metadata,
                "search_text": search_text,
                "vector": embed_text(search_text),
            }
        )
    return rows


def run_queries(db_dir: Path, collection_name: str, queries: list[str], n_results: int) -> dict[str, Any]:
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = get_collection(client, collection_name)
    rows = load_collection_rows(collection)

    results_summary: list[dict[str, Any]] = []
    for query in queries:
        query_vector = embed_text(query)
        scored_rows = []
        for row in rows:
            vector_score = cosine_similarity(query_vector, row["vector"])
            boost = lexical_boost(query, row["metadata"], row["document"])
            total_score = vector_score + boost
            scored_rows.append((total_score, vector_score, row))

        scored_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
        top_rows = scored_rows[:n_results]

        log(f"Query: {query}")
        if not top_rows:
            log("No results")
            results_summary.append({"query": query, "hit": False, "result_count": 0, "items": []})
            continue

        items: list[dict[str, Any]] = []
        for index, (total_score, vector_score, row) in enumerate(top_rows, start=1):
            snippet = summarize_document(row["document"])
            log(f"  {index}. id={row['id']}")
            log(f"     doc={snippet}")
            log(f"     metadata={json.dumps(row['metadata'], ensure_ascii=False)}")
            log(f"     score={round(total_score, 4)} vector={round(vector_score, 4)}")
            items.append(
                {
                    "id": row["id"],
                    "document": row["document"],
                    "document_summary": snippet,
                    "metadata": row["metadata"],
                    "score": total_score,
                    "vector_score": vector_score,
                }
            )

        results_summary.append(
            {
                "query": query,
                "hit": bool(items),
                "result_count": len(items),
                "items": items,
            }
        )

    return {
        "db_dir": str(db_dir.resolve()),
        "collection_name": collection_name,
        "query_count": len(queries),
        "n_results": n_results,
        "queries": results_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the GB 2760 A.1 local Chroma collection.")
    parser.add_argument("--db-dir", default=str(DEFAULT_DB_DIR), help="Persistent Chroma directory.")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION, help="Chroma collection name.")
    parser.add_argument("--n-results", type=int, default=3, help="Results returned for each query.")
    parser.add_argument("--summary-json", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES, help="Queries to run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_queries(
            db_dir=Path(args.db_dir),
            collection_name=args.collection_name,
            queries=args.queries,
            n_results=max(1, int(args.n_results)),
        )

        if args.summary_json:
            summary_path = Path(args.summary_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return 0
    except Exception as error:  # noqa: BLE001
        log(f"Fatal error: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
