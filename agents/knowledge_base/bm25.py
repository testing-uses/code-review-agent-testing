"""
agents/knowledge_base/bm25.py

Pure-Python BM25 scoring — the lexical half of the hybrid retrieval layer.
Complements the cosine-similarity term-vector search in kb_query.py: BM25
catches exact identifier/keyword matches (e.g. "borrow_book") that a
frequency-only cosine score can under-weight, while cosine catches
semantic similarity BM25 misses.
"""

import math
import re
from collections import Counter
from typing import Dict, List

K1 = 1.5
B = 0.75


def tokenize(text: str) -> List[str]:
    raw_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
    tokens = []
    for tok in raw_tokens:
        tok_lower = tok.lower()
        if len(tok_lower) > 2:
            tokens.append(tok_lower)
        # Split camelCase and snake_case sub-tokens for keyword matching
        sub_tokens = re.findall(r"[a-z]+|[A-Z][a-z]*|\d+", tok)
        for st in sub_tokens:
            st_lower = st.lower()
            if len(st_lower) > 2 and st_lower != tok_lower:
                tokens.append(st_lower)
    return tokens


def compute_bm25_scores(query_text: str, documents: Dict[str, str]) -> Dict[str, float]:
    """documents: {doc_id: document_text}. Returns {doc_id: normalized_score}."""
    query_terms = set(tokenize(query_text))
    if not documents or not query_terms:
        return {doc_id: 0.0 for doc_id in documents}

    doc_tokens = {doc_id: tokenize(text) for doc_id, text in documents.items()}
    doc_lengths = {doc_id: len(tokens) for doc_id, tokens in doc_tokens.items()}
    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 1.0
    num_docs = len(documents)

    doc_frequency = Counter()
    for tokens in doc_tokens.values():
        present_terms = set(tokens) & query_terms
        for term in present_terms:
            doc_frequency[term] += 1

    idf = {
        term: math.log((num_docs - doc_frequency[term] + 0.5) / (doc_frequency[term] + 0.5) + 1)
        for term in query_terms
    }

    raw_scores: Dict[str, float] = {}
    for doc_id, tokens in doc_tokens.items():
        term_freq = Counter(tokens)
        doc_length = doc_lengths[doc_id] or 1
        score = 0.0
        for term in query_terms:
            if term not in term_freq:
                continue
            frequency = term_freq[term]
            numerator = frequency * (K1 + 1)
            denominator = frequency + K1 * (1 - B + B * doc_length / avg_doc_length)
            score += idf[term] * (numerator / denominator)
        raw_scores[doc_id] = score

    max_score = max(raw_scores.values()) if raw_scores else 0.0
    if max_score == 0:
        return {doc_id: 0.0 for doc_id in raw_scores}
    return {doc_id: score / max_score for doc_id, score in raw_scores.items()}
