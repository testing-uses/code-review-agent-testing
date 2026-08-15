"""
agents/knowledge_base/query.py

Read-side API of the central knowledge base, used by every agent
(Dev Agent, Code Review Agent, future QA Agent) instead of each agent
rebuilding its own context from scratch.

CHANGES in this version:
  - personalized_pagerank(): fixed a dangling-node mass leak. Any node
    with no outgoing edges (very common for leaf functions) was dropping
    its rank each iteration instead of redistributing it, so total rank
    mass shrank every round instead of staying conserved at 1.0 — this
    was silently under-weighting anything mostly reached via leaf calls
    in context_selector's fused relevance score. Standard PPR fix:
    collect dangling mass, re-inject it through the personalization
    vector each iteration.
  - get_signatures_excluding(): added so context_builder.py can resolve
    symbol signatures from THIS central KB instead of maintaining a
    second, disconnected symbol cache (repo_map.py). One knowledge base,
    shared by dev agent and review agent alike.
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


def get_signatures_excluding(conn: sqlite3.Connection, names: Set[str], exclude_files: Set[str]) -> List[dict]:
    """Same as get_signatures, but drops symbols defined in files already
    covered elsewhere (e.g. already shown in full via the diff), and
    de-duplicates by symbol name — the same job repo_map.resolve_symbols
    used to do against its own separate cache. This is now the single
    place both agents resolve "what does this identifier refer to"."""
    if not names:
        return []
    results = get_signatures(conn, list(names))
    seen = set()
    filtered = []
    for entry in results:
        if entry["file"] in exclude_files:
            continue
        if entry["name"] in seen:
            continue
        seen.add(entry["name"])
        filtered.append(entry)
    return filtered


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
    actually being changed by the current task.

    Dangling-node fix: a node with no outgoing edges used to just drop its
    rank on the floor each iteration (`if not out_links: continue`), which
    leaks probability mass out of the system every round instead of
    conserving it at 1.0. That systematically under-scores anything mostly
    reached through leaf functions (getters, helpers with no further calls
    — i.e. a lot of real code). Fix: collect the dangling mass each round
    and redistribute it via the personalization vector, same as standard
    personalized PageRank / topic-sensitive PageRank formulations."""
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
    dangling_nodes = [node for node in nodes if not outgoing.get(node)]

    for _ in range(iterations):
        dangling_mass = damping * sum(rank[node] for node in dangling_nodes)

        new_rank = {
            node: (1 - damping) * personalization.get(node, 0.0)
                  + dangling_mass * personalization.get(node, 0.0)
            for node in nodes
        }
        for node in nodes:
            out_links = outgoing.get(node, [])
            if not out_links:
                continue
            share = damping * rank[node] / len(out_links)
            for target in out_links:
                new_rank[target] = new_rank.get(target, 0.0) + share
        rank = new_rank

    return dict(sorted(rank.items(), key=lambda item: item[1], reverse=True))