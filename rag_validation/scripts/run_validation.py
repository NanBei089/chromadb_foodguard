#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run validation tests for the GB 2760 A.1 Chroma collection."""

from __future__ import annotations

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb

from validation_common import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DB_DIR,
    DEFAULT_REPORT_OUTPUT,
    DEFAULT_RESULTS_OUTPUT,
    DEFAULT_TEST_QUERIES,
    category_matches_expected,
    collection_query,
    normalize_text,
    resolve_input_path,
    term_matches_expected,
    validate_record,
)

def log(message: str) -> None:
    print(f'[run_validation] {message}')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run RAG validation tests for GB 2760 A.1.')
    parser.add_argument('--input', default=None, help='Input JSONL path.')
    parser.add_argument('--queries', default=str(DEFAULT_TEST_QUERIES), help='Test query JSON path.')
    parser.add_argument('--db-dir', default=str(DEFAULT_DB_DIR), help='Validation Chroma db path.')
    parser.add_argument('--collection-name', default=DEFAULT_COLLECTION_NAME, help='Validation collection name.')
    parser.add_argument('--n-results', type=int, default=3, help='Top-k results.')
    parser.add_argument('--results-output', default=str(DEFAULT_RESULTS_OUTPUT), help='Validation results JSON output.')
    parser.add_argument('--report-output', default=str(DEFAULT_REPORT_OUTPUT), help='Validation Markdown report output.')
    return parser.parse_args()

def load_test_queries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('test_queries.json must contain a JSON array.')
    return data

def data_layer_summary(input_path: Path) -> dict[str, Any]:
    total_records = 0
    duplicate_id_count = 0
    duplicate_ids: list[str] = []
    empty_embedding_text_count = 0
    missing_field_counts = {field: 0 for field in ('id', 'term', 'normalized_term', 'function_category', 'embedding_text')}
    seen_ids: set[str] = set()

    with input_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total_records += 1
            record = json.loads(line)
            issues = validate_record(record)
            for field, is_missing in issues.items():
                if is_missing:
                    missing_field_counts[field] += 1
            if issues['embedding_text']:
                empty_embedding_text_count += 1
            record_id = normalize_text(record.get('id'))
            if record_id:
                if record_id in seen_ids:
                    duplicate_id_count += 1
                    duplicate_ids.append(record_id)
                else:
                    seen_ids.add(record_id)

    return {
        'total_records': total_records,
        'duplicate_id_count': duplicate_id_count,
        'duplicate_ids': duplicate_ids[:10],
        'empty_embedding_text_count': empty_embedding_text_count,
        'missing_field_counts': missing_field_counts,
    }

def evaluate_query(collection, case: dict[str, Any], n_results: int) -> dict[str, Any]:
    results = collection_query(
        collection=collection,
        query=case['query'],
        query_type=case['query_type'],
        n_results=n_results,
        candidate_pool=max(10, n_results * 4),
    )

    top_ids = [item['id'] for item in results]
    top_terms = [normalize_text(item['metadata'].get('term')) for item in results]
    top_normalized_terms = [normalize_text(item['metadata'].get('normalized_term')) for item in results]
    top_function_categories = [normalize_text(item['metadata'].get('function_category')) for item in results]
    top_documents = [normalize_text(item['document']) for item in results]

    merged_terms = top_terms + top_normalized_terms
    term_hit = term_matches_expected(merged_terms, case.get('expected_terms', []))
    category_hit = category_matches_expected(top_function_categories, case.get('expected_function_category'))

    top1_term_hit = False
    if results:
        top1_terms = [top_terms[0], top_normalized_terms[0]]
        top1_term_hit = term_matches_expected(top1_terms, case.get('expected_terms', []))

    return {
        'id': case['id'],
        'query': case['query'],
        'query_type': case['query_type'],
        'expected_terms': case.get('expected_terms', []),
        'expected_function_category': case.get('expected_function_category'),
        'notes': case.get('notes', ''),
        'top3_ids': top_ids,
        'top3_terms': top_terms,
        'top3_normalized_terms': top_normalized_terms,
        'top3_function_categories': top_function_categories,
        'top3_documents': top_documents,
        'term_hit': term_hit,
        'category_hit': category_hit,
        'top1_term_hit': top1_term_hit,
        'top3_term_hit': term_hit,
        'result_count': len(results),
        'results': results,
    }

def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_tests = len(results)
    term_hit_count = sum(1 for item in results if item['term_hit'])
    category_hit_count = sum(1 for item in results if item['category_hit'])
    top1_term_hit_count = sum(1 for item in results if item['top1_term_hit'])
    top3_term_hit_count = sum(1 for item in results if item['top3_term_hit'])

    by_query_type: dict[str, dict[str, Any]] = defaultdict(lambda: {
        'count': 0,
        'term_hit_count': 0,
        'category_hit_count': 0,
        'top1_term_hit_count': 0,
        'top3_term_hit_count': 0,
    })
    for item in results:
        bucket = by_query_type[item['query_type']]
        bucket['count'] += 1
        bucket['term_hit_count'] += int(item['term_hit'])
        bucket['category_hit_count'] += int(item['category_hit'])
        bucket['top1_term_hit_count'] += int(item['top1_term_hit'])
        bucket['top3_term_hit_count'] += int(item['top3_term_hit'])

    by_query_type_summary: dict[str, dict[str, Any]] = {}
    for query_type, bucket in by_query_type.items():
        count = bucket['count'] or 1
        by_query_type_summary[query_type] = {
            **bucket,
            'term_hit_rate': round(bucket['term_hit_count'] / count, 4),
            'category_hit_rate': round(bucket['category_hit_count'] / count, 4),
            'top1_term_hit_rate': round(bucket['top1_term_hit_count'] / count, 4),
            'top3_term_hit_rate': round(bucket['top3_term_hit_count'] / count, 4),
        }

    worst_query_type = min(
        by_query_type_summary.items(),
        key=lambda item: (item[1]['top3_term_hit_rate'], item[1]['category_hit_rate'])
    )[0] if by_query_type_summary else ''

    return {
        'total_tests': total_tests,
        'term_hit_count': term_hit_count,
        'category_hit_count': category_hit_count,
        'top1_term_hit_count': top1_term_hit_count,
        'top3_term_hit_count': top3_term_hit_count,
        'term_hit_rate': round(term_hit_count / total_tests, 4) if total_tests else 0.0,
        'category_hit_rate': round(category_hit_count / total_tests, 4) if total_tests else 0.0,
        'top1_term_hit_rate': round(top1_term_hit_count / total_tests, 4) if total_tests else 0.0,
        'top3_term_hit_rate': round(top3_term_hit_count / total_tests, 4) if total_tests else 0.0,
        'by_query_type': by_query_type_summary,
        'worst_query_type': worst_query_type,
    }

def build_problem_analysis(summary: dict[str, Any], failures: list[dict[str, Any]], data_summary: dict[str, Any]) -> list[str]:
    analysis: list[str] = []
    if data_summary['duplicate_id_count']:
        analysis.append(f"输入 JSONL 中存在 {data_summary['duplicate_id_count']} 个重复 id，需要在正式链路继续防重。")
    if data_summary['empty_embedding_text_count']:
        analysis.append(f"存在 {data_summary['empty_embedding_text_count']} 条空 embedding_text，会直接削弱向量检索稳定性。")

    worst_query_type = summary.get('worst_query_type')
    if worst_query_type == 'category_query':
        analysis.append('类别型查询表现最弱，原因是当前最小知识单元仍是一个添加剂一条记录，更适合实体检索，不擅长直接返回完整类别枚举。')
    if worst_query_type == 'parallel_term':
        analysis.append('并列术语查询表现偏弱，主要因为单个组名同时覆盖多个化学名，部分拆分词只能依赖 keywords 与 rerank。')
    if worst_query_type == 'alias_query':
        analysis.append('别名查询表现偏弱，短别名在向量空间中区分度有限，仍然依赖 aliases_text 和关键词增益。')
    if worst_query_type == 'rule_query':
        analysis.append('规则型查询表现偏弱，原因是 embedding_text 只保留代表性食品类别，复杂规则细节会被压缩。')

    if failures:
        analysis.append('失败样例主要集中在 top1 排序偏移和类别枚举问题，说明当前检索更适合作为召回层，而不是直接承担最终生成层答案。')
    else:
        analysis.append('当前测试未发现明显失败样例，数据层和索引层状态稳定。')
    return analysis

def build_markdown_report(
    input_path: Path,
    collection_name: str,
    collection_count: int,
    test_count: int,
    summary: dict[str, Any],
    data_summary: dict[str, Any],
    failures: list[dict[str, Any]],
    problem_analysis: list[str],
) -> str:
    by_type_lines = []
    for query_type, metrics in summary['by_query_type'].items():
        by_type_lines.append(
            f"- {query_type}: top1={metrics['top1_term_hit_rate']:.2%}, top3={metrics['top3_term_hit_rate']:.2%}, category={metrics['category_hit_rate']:.2%}, count={metrics['count']}"
        )
    failure_lines = []
    for item in failures[:8]:
        failure_lines.append(
            f"- {item['query']} ({item['query_type']}): top3={item['top3_normalized_terms']} / category={item['top3_function_categories']} / term_hit={item['term_hit']} / category_hit={item['category_hit']}"
        )
    if not failure_lines:
        failure_lines.append('- 无明显失败样例。')

    recommendation = '建议进入下一阶段（正式 RAG 问答集成）。' if summary['top3_term_hit_rate'] >= 0.8 and collection_count > 0 else '暂不建议直接进入下一阶段，需先优化检索排序或类别型问法支持。'

    lines = [
        '# Validation Report',
        '',
        '## Overview',
        f"- 输入数据文件路径: {input_path.resolve()}",
        f"- collection 名称: {collection_name}",
        f"- collection 数据量: {collection_count}",
        f"- 测试集数量: {test_count}",
        '',
        '## Data Layer Validation',
        f"- 空 embedding_text 数量: {data_summary['empty_embedding_text_count']}",
        f"- 重复 id 数量: {data_summary['duplicate_id_count']}",
        f"- 必要字段缺失统计: {json.dumps(data_summary['missing_field_counts'], ensure_ascii=False)}",
        '',
        '## Retrieval Metrics',
        f"- 总体术语命中率: {summary['term_hit_rate']:.2%}",
        f"- top1 术语命中率: {summary['top1_term_hit_rate']:.2%}",
        f"- top3 术语命中率: {summary['top3_term_hit_rate']:.2%}",
        f"- 类别命中率: {summary['category_hit_rate']:.2%}",
        '',
        '## Query Type Breakdown',
        *by_type_lines,
        '',
        '## Failed Examples',
        *failure_lines,
        '',
        '## Problem Analysis',
        *[f"- {line}" for line in problem_analysis],
        '',
        '## Recommendation',
        f"- {recommendation}",
    ]
    return '\n'.join(lines) + '\n'

def main() -> int:
    args = parse_args()
    try:
        input_path = resolve_input_path(args.input)
        queries_path = Path(args.queries)
        if not queries_path.is_absolute():
            queries_path = (Path(__file__).resolve().parents[1] / queries_path).resolve()
        db_dir = Path(args.db_dir)
        if not db_dir.is_absolute():
            db_dir = (Path(__file__).resolve().parents[1] / db_dir).resolve()
        results_output = Path(args.results_output)
        if not results_output.is_absolute():
            results_output = (Path(__file__).resolve().parents[1] / results_output).resolve()
        report_output = Path(args.report_output)
        if not report_output.is_absolute():
            report_output = (Path(__file__).resolve().parents[1] / report_output).resolve()
        results_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.parent.mkdir(parents=True, exist_ok=True)

        data_summary = data_layer_summary(input_path)
        test_queries = load_test_queries(queries_path)
        if not test_queries:
            raise ValueError('Test query set is empty.')

        client = chromadb.PersistentClient(path=str(db_dir))
        try:
            collection = client.get_collection(name=args.collection_name)
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f'Collection not found: {args.collection_name}. Please run build_test_chroma.py first.') from error

        collection_count = collection.count()
        if collection_count <= 0:
            raise RuntimeError('Collection exists but is empty.')

        results = [evaluate_query(collection, case, n_results=max(1, int(args.n_results))) for case in test_queries]
        summary = summarize_results(results)
        failures = [item for item in results if not item['term_hit'] or not item['category_hit']]
        analysis = build_problem_analysis(summary, failures, data_summary)

        payload = {
            'input_file': str(input_path.resolve()),
            'collection_name': args.collection_name,
            'collection_count': collection_count,
            'test_query_count': len(test_queries),
            'data_layer': data_summary,
            'summary': summary,
            'results': results,
        }
        results_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        report_output.write_text(
            build_markdown_report(
                input_path=input_path,
                collection_name=args.collection_name,
                collection_count=collection_count,
                test_count=len(test_queries),
                summary=summary,
                data_summary=data_summary,
                failures=failures,
                problem_analysis=analysis,
            ),
            encoding='utf-8',
        )

        log(f'collection 数据量: {collection_count}')
        log(f'测试集数量: {len(test_queries)}')
        log(f"top1 命中率: {summary['top1_term_hit_rate']:.2%}")
        log(f"top3 命中率: {summary['top3_term_hit_rate']:.2%}")
        log(f"最弱查询类型: {summary['worst_query_type']}")
        return 0
    except Exception as error:  # noqa: BLE001
        log(f'Fatal error: {error}')
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
