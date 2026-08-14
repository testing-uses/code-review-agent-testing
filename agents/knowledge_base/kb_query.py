"""
agents/knowledge_base/query.py

Read-side API of the central knowledge base, used by every agent
(Dev Agent, Code Review Agent, future QA Agent) instead of each agent
rebuilding its own context from scratch.

Implements, in pure Python (no extra dependencies):
    - reverse_dependencies(): the exact check that would have caught the
      `Loan` removal bug deterministically.
    - keyword_search(): cosine similarity over term-frequency vectors —
      the lightweight stand-in for vector-DB semantic search.
    - personalized_pagerank(): manual power-iteration PageRank seeded on
      the files/symbols actually touched by the current task, matching
      the same algorithm used by Aider's repo map.
"""

import json
import math
import sqlite3
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from kb_schema import get_connection


def reverse_dependencies(conn: sqlite3.Connection, symbol_name: str) -> List[Tuple[str, str]]:
    """Who calls/imports this symbol? Returns [(src_symbol, src_file), ...]."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT src_symbol, src_file FROM edges WHERE dst_symbol = ?",
        (symbol_name,),
    )
    return cursor.fetchall()


def get_signatures(conn: sqlite3.Connection, names: List[str]) -> List[dict]:
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in names)
    if not names:
        return []
    cursor.execute(
        f"SELECT name, file, kind, signature, docstring FROM symbols WHERE name IN ({placeholders})",
        names,
    )
    return [
        {"name": r[0], "file": r[1], "kind": r[2], "signature": r[3], "docstring": r[4]}
        for r in cursor.fetchall()
    ]


def _cosine_similarity(vec_a: Dict[str, int], vec_b: Dict[str, int]) -> float:
    common_terms = set(vec_a) & set(vec_b)
    if not common_terms:
        return 0.0
    dot_product = sum(vec_a[t] * vec_b[t] for t in common_terms)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def keyword_search(conn: sqlite3.Connection, query_text: str, top_k: int = 5) -> List[dict]:
    """Cosine-similarity search over term-frequency vectors.
    This is the "vector-DB-like" retrieval layer for the POC."""
    import re
    from collections import Counter

    query_terms = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query_text.lower())
    query_vector = dict(Counter(t for t in query_terms if len(t) > 2))

    cursor = conn.cursor()
    cursor.execute("SELECT name, file, kind, signature, docstring, term_vector FROM symbols")

    scored = []
    for name, file_path, kind, signature, docstring, vector_json in cursor.fetchall():
        symbol_vector = json.loads(vector_json) if vector_json else {}
        score = _cosine_similarity(query_vector, symbol_vector)
        if score > 0:
            scored.append({
                "name": name, "file": file_path, "kind": kind,
                "signature": signature, "docstring": docstring, "score": score,
            })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def personalized_pagerank(
    conn: sqlite3.Connection,
    seed_symbols: Set[str],
    damping: float = 0.85,
    iterations: int = 30,
) -> Dict[str, float]:
    """Manual power-iteration PageRank over the symbol reference graph,
    personalized (restart-biased) toward the seed symbols — i.e. the files
    actually being changed by the current task. Same algorithm family as
    Aider's repo map ranking, implemented without a graph-library dependency."""
    cursor = conn.cursor()
    cursor.execute("SELECT src_symbol, dst_symbol FROM edges")
    edges = cursor.fetchall()

    nodes: Set[str] = set()
    outgoing: Dict[str, List[str]] = defaultdict(list)
    for src, dst in edges:
        nodes.add(src)
        nodes.add(dst)
        outgoing[src].append(dst)

    nodes.update(seed_symbols)
    if not nodes:
        return {}

    personalization = {node: (1.0 if node in seed_symbols else 0.0) for node in nodes}
    total_seed_weight = sum(personalization.values()) or 1.0
    personalization = {node: weight / total_seed_weight for node, weight in personalization.items()}

    rank = {node: 1.0 / len(nodes) for node in nodes}

    for _ in range(iterations):
        new_rank = {node: (1 - damping) * personalization.get(node, 0.0) for node in nodes}
        for node in nodes:
            out_links = outgoing.get(node, [])
            if not out_links:
                continue
            share = damping * rank[node] / len(out_links)
            for target in out_links:
                new_rank[target] = new_rank.get(target, 0.0) + share
        rank = new_rank

    return dict(sorted(rank.items(), key=lambda item: item[1], reverse=True))
