#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ingest GB 2760-2024 A.1 grouped JSONL records into a local Chroma collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import traceback
import unicodedata
from pathlib import Path
from typing import Any

import chromadb
import httpx

DEFAULT_INPUT_CANDIDATES = [
    Path("data/processed/gb2760_a1_grouped_min_final_prod.jsonl"),
    Path("data/processed/gb2760_a1_grouped_min_final_v2.jsonl"),
    Path("data/processed/gb2760_a1_grouped_min_final.jsonl"),
]
DEFAULT_DB_DIR = Path("chroma_db")
DEFAULT_COLLECTION = "gb2760_a1_grouped"
DEFAULT_BATCH_SIZE = 100
VECTOR_DIM = 512
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3-embedding:latest"


def log(message: str) -> None:
    print(f"[ingest] {message}")


def resolve_input_path(cli_input: str | None) -> Path:
    if cli_input:
        path = Path(cli_input)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    matches = sorted(Path("data/processed").glob("*grouped*prod*.jsonl"))
    if not matches:
        matches = sorted(Path("data/processed").glob("*grouped*final*.jsonl"))
    if not matches:
        matches = sorted(Path("data/processed").glob("*grouped*.jsonl"))
    if matches:
        return matches[0]

    raise FileNotFoundError("No suitable grouped JSONL input file was found.")


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
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


def _resolve_ollama_settings(base_url: str | None, model: str | None) -> tuple[str, str]:
    resolved_base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    resolved_model = model or os.getenv("OLLAMA_EMBEDDING_MODEL") or DEFAULT_OLLAMA_MODEL
    return resolved_base_url, resolved_model


def embed_texts_ollama(texts: list[str], base_url: str, model: str, timeout_s: float = 60.0) -> list[list[float]]:
    endpoint = f"{base_url.rstrip('/')}/api/embed"
    payload = {
        "model": model,
        "input": texts,
        "truncate": True,
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError("Ollama embedding request failed") from exc
    except ValueError as exc:
        raise RuntimeError("Ollama embedding response is not valid JSON") from exc

    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise RuntimeError("Ollama embedding response size mismatch")

    vectors: list[list[float]] = []
    for vector in embeddings:
        if not isinstance(vector, list) or not vector:
            raise RuntimeError("Ollama embedding response contains an invalid vector")
        try:
            vectors.append([float(item) for item in vector])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Ollama embedding vector contains non-numeric values") from exc
    return vectors


def clean_keywords(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = normalize_text(item)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def build_metadata(record: dict[str, Any]) -> dict[str, Any]:
    aliases = record.get("aliases")
    alias_list = aliases if isinstance(aliases, list) else []
    cleaned_aliases: list[str] = []
    seen_aliases: set[str] = set()
    for item in alias_list:
        text = normalize_text(item)
        if text and text not in seen_aliases:
            seen_aliases.add(text)
            cleaned_aliases.append(text)

    keywords = clean_keywords(record.get("keywords"))
    rules = record.get("rules")
    rule_count = len(rules) if isinstance(rules, list) else 0

    metadata: dict[str, Any] = {
        "term": normalize_text(record.get("term")),
        "normalized_term": normalize_text(record.get("normalized_term")),
        "function_category": normalize_text(record.get("function_category")),
        "rule_count": rule_count,
        "has_alias": bool(cleaned_aliases),
        "aliases_text": "；".join(cleaned_aliases) if cleaned_aliases else "",
    }
    if keywords:
        metadata["keywords"] = keywords
    return metadata


def build_embedding_source(record: dict[str, Any], metadata: dict[str, Any], document: str) -> str:
    parts: list[str] = []
    term = normalize_text(record.get("term"))
    normalized_term = normalize_text(record.get("normalized_term"))
    if term:
        parts.extend([term, term])
    if normalized_term:
        parts.extend([normalized_term, normalized_term])

    aliases = record.get("aliases") if isinstance(record.get("aliases"), list) else []
    for alias in aliases:
        alias_text = normalize_text(alias)
        if alias_text:
            parts.extend([alias_text, alias_text])

    function_category = metadata.get("function_category")
    if isinstance(function_category, str) and function_category:
        parts.append(function_category)

    keywords = metadata.get("keywords")
    if isinstance(keywords, list):
        parts.extend(normalize_text(item) for item in keywords if normalize_text(item))

    parts.append(document)
    return " | ".join(part for part in parts if part)


def clear_collection(collection: Any, page_size: int = 1000) -> int:
    existing = collection.count()
    if existing <= 0:
        return 0

    deleted = 0
    while True:
        batch = collection.get(limit=page_size)
        ids = batch.get("ids") or []
        if not ids:
            break
        collection.delete(ids=ids)
        deleted += len(ids)
        if len(ids) < page_size:
            break
    return deleted


def upsert_batch(
    collection: Any,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> tuple[int, int]:
    if not ids:
        return 0, 0

    try:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        return len(ids), 0
    except Exception as batch_error:  # noqa: BLE001
        log(f"Batch upsert failed for {len(ids)} records, fallback to single upserts: {batch_error}")

    success = 0
    skipped = 0
    for record_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
        try:
            collection.upsert(
                ids=[record_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[embedding],
            )
            success += 1
        except Exception as record_error:  # noqa: BLE001
            skipped += 1
            log(f"Skip failed record {record_id}: {record_error}")
    return success, skipped


def ingest_records(
    input_path: Path,
    db_dir: Path,
    collection_name: str,
    batch_size: int,
    embedding: str,
    ollama_base_url: str | None,
    ollama_model: str | None,
    reset_collection: bool,
) -> dict[str, Any]:
    db_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_dir))
    if reset_collection:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(name=collection_name)
    deleted_existing = clear_collection(collection)
    if deleted_existing:
        log(f"Cleared {deleted_existing} existing records from collection {collection_name}")

    total_lines = 0
    ingested = 0
    skipped = 0
    skipped_missing = 0
    skipped_invalid_json = 0
    embedding_dimension = None
    resolved_ollama_base_url, resolved_ollama_model = _resolve_ollama_settings(ollama_base_url, ollama_model)
    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_metas: list[dict[str, Any]] = []
    batch_sources: list[str] = []
    batch_embeddings: list[list[float]] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            total_lines += 1
            line = raw_line.strip()
            if not line:
                skipped += 1
                skipped_missing += 1
                log(f"Skip blank line at {line_number}")
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                skipped += 1
                skipped_invalid_json += 1
                log(f"Skip invalid JSON at line {line_number}: {error}")
                continue

            record_id = normalize_text(record.get("id"))
            document = normalize_text(record.get("embedding_text"))
            if not record_id or not document:
                skipped += 1
                skipped_missing += 1
                log(f"Skip line {line_number}: missing id or embedding_text")
                continue

            metadata = build_metadata(record)
            embedding_source = build_embedding_source(record, metadata, document)
            batch_ids.append(record_id)
            batch_docs.append(document)
            batch_metas.append(metadata)
            batch_sources.append(embedding_source)

            if len(batch_ids) >= batch_size:
                if embedding == "ollama":
                    batch_embeddings = embed_texts_ollama(
                        batch_sources,
                        base_url=resolved_ollama_base_url,
                        model=resolved_ollama_model,
                    )
                else:
                    batch_embeddings = [embed_text(source) for source in batch_sources]
                if batch_embeddings and embedding_dimension is None:
                    embedding_dimension = len(batch_embeddings[0])
                success, batch_skipped = upsert_batch(collection, batch_ids, batch_docs, batch_metas, batch_embeddings)
                ingested += success
                skipped += batch_skipped
                batch_ids.clear()
                batch_docs.clear()
                batch_metas.clear()
                batch_sources.clear()
                batch_embeddings.clear()

    if batch_ids:
        if embedding == "ollama":
            batch_embeddings = embed_texts_ollama(
                batch_sources,
                base_url=resolved_ollama_base_url,
                model=resolved_ollama_model,
            )
        else:
            batch_embeddings = [embed_text(source) for source in batch_sources]
        if batch_embeddings and embedding_dimension is None:
            embedding_dimension = len(batch_embeddings[0])
        success, batch_skipped = upsert_batch(collection, batch_ids, batch_docs, batch_metas, batch_embeddings)
        ingested += success
        skipped += batch_skipped

    embedding_strategy = "hashed_ngrams(term+normalized_term+aliases+keywords+embedding_text)"
    if embedding == "ollama":
        embedding_strategy = f"ollama:{resolved_ollama_model}"

    return {
        "input_file": str(input_path.resolve()),
        "db_dir": str(db_dir.resolve()),
        "collection_name": collection_name,
        "total_lines": total_lines,
        "ingested_count": ingested,
        "skipped_count": skipped,
        "skipped_missing_or_blank": skipped_missing,
        "skipped_invalid_json": skipped_invalid_json,
        "collection_count": collection.count(),
        "deleted_existing_count": deleted_existing,
        "batch_size": batch_size,
        "embedding_strategy": embedding_strategy,
        "embedding_dimension": embedding_dimension,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest GB 2760 A.1 grouped JSONL into local Chroma.")
    parser.add_argument("--input", default=None, help="Input JSONL file path.")
    parser.add_argument("--db-dir", default=str(DEFAULT_DB_DIR), help="Persistent Chroma directory.")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION, help="Chroma collection name.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for Chroma upsert.")
    parser.add_argument("--embedding", choices=("hashed", "ollama"), default="hashed", help="Embedding mode.")
    parser.add_argument("--ollama-base-url", default=None, help="Ollama base URL, defaults to env OLLAMA_BASE_URL.")
    parser.add_argument("--ollama-model", default=None, help="Ollama embedding model, defaults to env OLLAMA_EMBEDDING_MODEL.")
    parser.add_argument("--reset-collection", action="store_true", help="Delete collection before ingest.")
    parser.add_argument("--summary-json", default=None, help="Optional JSON summary output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        input_path = resolve_input_path(args.input)
        summary = ingest_records(
            input_path=input_path,
            db_dir=Path(args.db_dir),
            collection_name=args.collection_name,
            batch_size=max(1, int(args.batch_size)),
            embedding=str(args.embedding),
            ollama_base_url=args.ollama_base_url,
            ollama_model=args.ollama_model,
            reset_collection=bool(args.reset_collection) or str(args.embedding) == "ollama",
        )

        if args.summary_json:
            summary_path = Path(args.summary_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        log(f"Collection: {summary['collection_name']}")
        log(f"Ingested: {summary['ingested_count']}")
        log(f"Skipped: {summary['skipped_count']}")
        log(f"DB dir: {summary['db_dir']}")
        return 0
    except Exception as error:  # noqa: BLE001
        log(f"Fatal error: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
