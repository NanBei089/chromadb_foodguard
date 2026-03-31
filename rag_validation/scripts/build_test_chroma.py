#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build an isolated Chroma validation collection for GB 2760 A.1."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import chromadb

from validation_common import (
    BATCH_SIZE,
    DEFAULT_BUILD_SUMMARY,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DB_DIR,
    build_embedding_source,
    build_metadata,
    embed_text,
    normalize_text,
    resolve_input_path,
    validate_record,
)

def log(message: str) -> None:
    print(f'[build_test_chroma] {message}')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build local validation Chroma collection.')
    parser.add_argument('--input', default=None, help='Input JSONL path.')
    parser.add_argument('--db-dir', default=str(DEFAULT_DB_DIR), help='Validation Chroma db directory.')
    parser.add_argument('--collection-name', default=DEFAULT_COLLECTION_NAME, help='Validation collection name.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size.')
    parser.add_argument('--summary-output', default=str(DEFAULT_BUILD_SUMMARY), help='Build summary JSON output.')
    return parser.parse_args()

def flush_batch(collection, ids, documents, metadatas, embeddings) -> tuple[int, int]:
    if not ids:
        return 0, 0
    try:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        return len(ids), 0
    except Exception as error:  # noqa: BLE001
        log(f'Batch upsert failed, fallback to single records: {error}')

    written = 0
    skipped = 0
    for record_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
        try:
            collection.upsert(ids=[record_id], documents=[document], metadatas=[metadata], embeddings=[embedding])
            written += 1
        except Exception as single_error:  # noqa: BLE001
            skipped += 1
            log(f'Skip failed record {record_id}: {single_error}')
    return written, skipped

def main() -> int:
    args = parse_args()
    try:
        input_path = resolve_input_path(args.input)
        db_dir = Path(args.db_dir)
        if not db_dir.is_absolute():
            db_dir = (Path(__file__).resolve().parents[1] / db_dir).resolve()
        db_dir.mkdir(parents=True, exist_ok=True)

        summary_path = Path(args.summary_output)
        if not summary_path.is_absolute():
            summary_path = (Path(__file__).resolve().parents[1] / summary_path).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(db_dir))
        try:
            client.delete_collection(args.collection_name)
            log(f'Deleted existing collection: {args.collection_name}')
        except Exception:  # noqa: BLE001
            pass
        collection = client.get_or_create_collection(name=args.collection_name)

        total_records = 0
        written_count = 0
        skipped_count = 0
        duplicate_id_count = 0
        invalid_json_count = 0
        missing_field_counts = {field: 0 for field in ('id', 'term', 'normalized_term', 'function_category', 'embedding_text')}
        seen_ids: set[str] = set()

        batch_ids: list[str] = []
        batch_docs: list[str] = []
        batch_metas: list[dict] = []
        batch_embeddings: list[list[float]] = []

        with input_path.open('r', encoding='utf-8') as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                total_records += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json_count += 1
                    skipped_count += 1
                    continue

                issues = validate_record(record)
                for field, is_missing in issues.items():
                    if is_missing:
                        missing_field_counts[field] += 1

                record_id = normalize_text(record.get('id'))
                document = normalize_text(record.get('embedding_text'))
                if not record_id or not document:
                    skipped_count += 1
                    continue
                if record_id in seen_ids:
                    duplicate_id_count += 1
                    skipped_count += 1
                    continue
                seen_ids.add(record_id)

                metadata = build_metadata(record)
                embedding_source = build_embedding_source(record, metadata, document)
                batch_ids.append(record_id)
                batch_docs.append(document)
                batch_metas.append(metadata)
                batch_embeddings.append(embed_text(embedding_source))

                if len(batch_ids) >= max(1, int(args.batch_size)):
                    written, skipped = flush_batch(collection, batch_ids, batch_docs, batch_metas, batch_embeddings)
                    written_count += written
                    skipped_count += skipped
                    batch_ids.clear()
                    batch_docs.clear()
                    batch_metas.clear()
                    batch_embeddings.clear()

        if batch_ids:
            written, skipped = flush_batch(collection, batch_ids, batch_docs, batch_metas, batch_embeddings)
            written_count += written
            skipped_count += skipped

        summary = {
            'input_file': str(input_path.resolve()),
            'collection_name': args.collection_name,
            'db_dir': str(db_dir.resolve()),
            'total_records': total_records,
            'written_count': written_count,
            'skipped_count': skipped_count,
            'collection_count': collection.count(),
            'duplicate_id_count': duplicate_id_count,
            'invalid_json_count': invalid_json_count,
            'missing_field_counts': missing_field_counts,
            'batch_size': max(1, int(args.batch_size)),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

        log(f"总记录数: {summary['total_records']}")
        log(f"成功写入数: {summary['written_count']}")
        log(f"跳过数: {summary['skipped_count']}")
        log(f"collection 名称: {summary['collection_name']}")
        log(f"数据库路径: {summary['db_dir']}")
        return 0
    except Exception as error:  # noqa: BLE001
        log(f'Fatal error: {error}')
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
