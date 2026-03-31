from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_ROOT = PROJECT_ROOT / 'rag_validation'
DEFAULT_INPUT_CANDIDATES = [
    PROJECT_ROOT / 'data/processed/gb2760_a1_grouped_min_final_prod.jsonl',
    PROJECT_ROOT / 'data/processed/gb2760_a1_grouped_min_final_v2.jsonl',
    PROJECT_ROOT / 'data/processed/gb2760_a1_grouped_min_final.jsonl',
]
DEFAULT_DB_DIR = VALIDATION_ROOT / 'chroma_db'
DEFAULT_COLLECTION_NAME = 'gb2760_a1_validation'
DEFAULT_TEST_QUERIES = VALIDATION_ROOT / 'data/test_queries.json'
DEFAULT_BUILD_SUMMARY = VALIDATION_ROOT / 'reports/build_summary.json'
DEFAULT_RESULTS_OUTPUT = VALIDATION_ROOT / 'reports/validation_results.json'
DEFAULT_REPORT_OUTPUT = VALIDATION_ROOT / 'reports/validation_report.md'
VECTOR_DIM = 512
BATCH_SIZE = 100
REQUIRED_FIELDS = ('id', 'term', 'normalized_term', 'function_category', 'embedding_text')
GENERIC_QUERY_PHRASES = [
    '可以用于哪些食品',
    '可用于哪些食品',
    '能用于哪些食品',
    '可以用于什么',
    '可用于什么',
    '能用于什么',
    '属于什么类别',
    '是什么类别',
    '是什么添加剂',
    '是什么',
    '有哪些',
    '有哪几种',
    '有哪类',
    '残留量',
]
STOP_TOKENS = {'是什么', '什么', '哪些', '有哪几种', '有哪些', '有哪类', '可以用于', '可用于', '能用于', '类别', '食品'}

def resolve_input_path(cli_input: str | None = None) -> Path:
    if cli_input:
        path = Path(cli_input)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f'Input file not found: {path}')
        return path
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError('No suitable input JSONL file was found.')

def normalize_text(value: Any) -> str:
    text = '' if value is None else str(value)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compact_text(value: Any) -> str:
    return normalize_text(value).replace(' ', '')

def canonical_text(value: Any) -> str:
    text = compact_text(value)
    text = text.replace('，', '、').replace(',', '、').replace('；', '、').replace(';', '、')
    text = text.replace('（', '(').replace('）', ')')
    return text.lower()

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

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def validate_record(record: dict[str, Any]) -> dict[str, bool]:
    issues: dict[str, bool] = {}
    for field in REQUIRED_FIELDS:
        issues[field] = not bool(normalize_text(record.get(field)))
    return issues

def build_metadata(record: dict[str, Any]) -> dict[str, Any]:
    aliases = record.get('aliases') if isinstance(record.get('aliases'), list) else []
    aliases_clean = []
    seen_aliases: set[str] = set()
    for item in aliases:
        text = normalize_text(item)
        if text and text not in seen_aliases:
            seen_aliases.add(text)
            aliases_clean.append(text)

    rules = record.get('rules') if isinstance(record.get('rules'), list) else []
    metadata: dict[str, Any] = {
        'term': normalize_text(record.get('term')),
        'normalized_term': normalize_text(record.get('normalized_term')),
        'function_category': normalize_text(record.get('function_category')),
        'rule_count': len(rules),
        'has_alias': bool(aliases_clean),
        'aliases_text': '；'.join(aliases_clean) if aliases_clean else '',
    }
    keywords = clean_keywords(record.get('keywords'))
    if keywords:
        metadata['keywords'] = keywords
    return metadata

def build_search_text(metadata: dict[str, Any], document: str) -> str:
    parts: list[str] = []
    for field in ('term', 'normalized_term', 'aliases_text', 'function_category'):
        text = normalize_text(metadata.get(field))
        if text:
            parts.extend([text, text])
    keywords = metadata.get('keywords')
    if isinstance(keywords, list):
        parts.extend(normalize_text(item) for item in keywords if normalize_text(item))
    parts.append(normalize_text(document))
    return ' | '.join(part for part in parts if part)

def build_embedding_source(record: dict[str, Any], metadata: dict[str, Any], document: str) -> str:
    parts: list[str] = []
    term = normalize_text(record.get('term'))
    normalized_term = normalize_text(record.get('normalized_term'))
    if term:
        parts.extend([term, term])
    if normalized_term:
        parts.extend([normalized_term, normalized_term])
    aliases = record.get('aliases') if isinstance(record.get('aliases'), list) else []
    for alias in aliases:
        alias_text = normalize_text(alias)
        if alias_text:
            parts.extend([alias_text, alias_text])
    keywords = metadata.get('keywords')
    if isinstance(keywords, list):
        parts.extend(normalize_text(item) for item in keywords if normalize_text(item))
    function_category = normalize_text(metadata.get('function_category'))
    if function_category:
        parts.append(function_category)
    parts.append(document)
    return ' | '.join(part for part in parts if part)

def feature_tokens(text: str) -> list[tuple[str, float]]:
    clean_text = normalize_text(text)
    if not clean_text:
        return []
    compact = clean_text.replace(' ', '')
    tokens: list[tuple[str, float]] = []
    for ch in compact:
        tokens.append((f'c:{ch}', 1.0))
    for size, weight in ((2, 1.9), (3, 2.5)):
        if len(compact) >= size:
            for index in range(len(compact) - size + 1):
                tokens.append((f'n{size}:{compact[index:index+size]}', weight))
    for word in re.findall(r'[\u4e00-\u9fffA-Za-z0-9.+≤≥/%-]+', clean_text):
        tokens.append((f'w:{word}', 2.2))
    return tokens

def embed_text(text: str, dim: int = VECTOR_DIM) -> list[float]:
    vector = [0.0] * dim
    for token, weight in feature_tokens(text):
        digest = hashlib.md5(token.encode('utf-8')).digest()
        index = int.from_bytes(digest[:4], 'big') % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]

def cosine_similarity(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))

def split_candidates(*values: Any) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_text(value)
        if not text:
            continue
        parts = re.split(r'[、,，;；/（）()]+', text)
        for part in [text, *parts]:
            candidate = normalize_text(part)
            if len(candidate) >= 2 and candidate not in seen:
                seen.add(candidate)
                output.append(candidate)
    return output

def strip_generic_phrases(query: str) -> str:
    text = normalize_text(query)
    for phrase in GENERIC_QUERY_PHRASES:
        text = text.replace(phrase, ' ')
    text = re.sub(r'\s+', ' ', text).strip(' ，,。？?、')
    return text or normalize_text(query)

def query_keywords(query: str) -> list[str]:
    base = strip_generic_phrases(query)
    tokens = [base, *re.findall(r'[\u4e00-\u9fffA-Za-z0-9.+≤≥/%-]+', base)]
    output: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        canonical = canonical_text(token)
        if canonical and canonical not in STOP_TOKENS and len(canonical) >= 2 and canonical not in seen:
            seen.add(canonical)
            output.append(canonical)
    return output

def term_matches_expected(found_terms: list[str], expected_terms: Any) -> bool:
    expected_list = expected_terms if isinstance(expected_terms, list) else [expected_terms]
    found = [canonical_text(item) for item in found_terms if canonical_text(item)]
    for expected in expected_list:
        expected_norm = canonical_text(expected)
        if not expected_norm:
            continue
        for candidate in found:
            if candidate == expected_norm or candidate in expected_norm or expected_norm in candidate:
                return True
    return False

def category_matches_expected(found_categories: list[str], expected_category: Any) -> bool:
    expected_list = expected_category if isinstance(expected_category, list) else [expected_category]
    found = [canonical_text(item) for item in found_categories if canonical_text(item)]
    for expected in expected_list:
        expected_norm = canonical_text(expected)
        if not expected_norm:
            continue
        for candidate in found:
            if candidate == expected_norm or candidate in expected_norm or expected_norm in candidate:
                return True
    return False

def lexical_boost(query: str, query_type: str, metadata: dict[str, Any], document: str) -> float:
    query_core = canonical_text(strip_generic_phrases(query))
    if not query_core:
        return 0.0

    boost = 0.0
    term = normalize_text(metadata.get('term'))
    normalized_term = normalize_text(metadata.get('normalized_term'))
    aliases = split_candidates(metadata.get('aliases_text'))
    keywords = metadata.get('keywords') if isinstance(metadata.get('keywords'), list) else []
    function_category = normalize_text(metadata.get('function_category'))
    function_category_c = canonical_text(function_category)
    document_c = canonical_text(document)

    core_candidates = split_candidates(term, normalized_term, *aliases)
    keyword_candidates = [candidate for candidate in split_candidates(*keywords) if len(canonical_text(candidate)) >= 4]

    for candidate in core_candidates:
        candidate_c = canonical_text(candidate)
        if candidate_c == query_core:
            boost += 24.0
        elif candidate_c and (candidate_c in query_core or query_core in candidate_c):
            boost += 18.0

    for candidate in keyword_candidates:
        candidate_c = canonical_text(candidate)
        if candidate_c == query_core:
            boost += 10.0
        elif candidate_c and (candidate_c in query_core or query_core in candidate_c):
            boost += 5.0

    if query_type == 'category_query' and query_core and query_core in function_category_c:
        boost += 22.0
    elif '类别' in normalize_text(query) and query_core and query_core in function_category_c:
        boost += 4.0

    for token in query_keywords(query):
        if token in function_category_c:
            boost += 2.0
        if token in document_c:
            boost += 0.8

    return boost

def collection_query(collection: Any, query: str, query_type: str, n_results: int = 3, candidate_pool: int = 10) -> list[dict[str, Any]]:
    base_query = strip_generic_phrases(query)
    raw = collection.query(
        query_embeddings=[embed_text(base_query)],
        n_results=max(n_results, candidate_pool),
        include=['documents', 'metadatas'],
    )
    ids = (raw.get('ids') or [[]])[0]
    documents = (raw.get('documents') or [[]])[0]
    metadatas = (raw.get('metadatas') or [[]])[0]

    query_vector = embed_text(base_query)
    scored: list[dict[str, Any]] = []
    for index, record_id in enumerate(ids):
        document = documents[index] if index < len(documents) else ''
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        search_text = build_search_text(metadata, document)
        vector_score = cosine_similarity(query_vector, embed_text(search_text))
        score = vector_score + lexical_boost(query, query_type, metadata, document)
        scored.append({
            'id': record_id,
            'document': document,
            'metadata': metadata,
            'score': score,
            'vector_score': vector_score,
        })

    scored.sort(key=lambda item: (item['score'], item['vector_score']), reverse=True)
    return scored[:n_results]
