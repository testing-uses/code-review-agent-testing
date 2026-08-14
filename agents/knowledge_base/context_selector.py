"""
agents/knowledge_base/context_selector.py

The unified context-assembly service used by BOTH the Dev Agent and the
Code Review Agent. Implements all three algorithms discussed:

    1. Hybrid retrieval  = BM25 (lexical) + cosine similarity (semantic).
    2. Personalized PageRank = ranks symbols by graph proximity to the
       task's seed files/symbols (same algorithm family as Aider's repo map).
    3. Knapsack selection = greedy by (fused_relevance / token_cost),
       fit under a hard DCBA-allocated token budget.

This replaces ad hoc keyword_search() calls scattered per-agent with one
auditable, reusable relevance-ranking pipeline.
"""

import json
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Set

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))

from bm25 import compute_bm25_scores  # noqa: E402
from kb_query import _cosine_similarity, personalized_pagerank  # noqa: E402
from groq_client import estimate_tokens  # noqa: E402

FUSION_WEIGHTS = {"bm25": 0.4, "cosine": 0.35, "pagerank": 0.25}


@dataclass
class ContextEntry:
    name: str
    file: str
    signature: str
    docstring: str
    fused_score: float
    token_cost: int


def _extract_query_term_vector(query_text: str) -> Dict[str, int]:
    import re
    from collections import Counter
    terms = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query_text.lower())
    return dict(Counter(t for t in terms if len(t) > 2))


def _identify_seed_symbols(query_text: str, all_symbol_names: Set[str]) -> Set[str]:
    """Symbols explicitly named in the task/diff text — used as PageRank
    restart seeds, i.e. "what the task is actually about"."""
    query_tokens = set(_extract_query_term_vector(query_text).keys())
    return query_tokens & all_symbol_names


def select_context(conn, query_text: str, budget_tokens: int, top_k_candidates: int = 30) -> str:
    """Returns a rendered, budget-fitted context string ready to drop into
    an agent's user prompt."""
    cursor = conn.cursor()
    cursor.execute("SELECT name, file, kind, signature, docstring, term_vector FROM symbols")
    rows = cursor.fetchall()

    if not rows:
        return "(knowledge base is empty — run build_kb.py first)"

    symbols = [
        {
            "name": r[0], "file": r[1], "kind": r[2], "signature": r[3],
            "docstring": r[4] or "", "term_vector": json.loads(r[5]) if r[5] else {},
        }
        for r in rows
    ]

    # ---- 1. Hybrid retrieval: BM25 (lexical) ----
    documents = {
        f"{s['name']}::{s['file']}": f"{s['name']} {s['signature']} {s['docstring']}"
        for s in symbols
    }
    bm25_scores = compute_bm25_scores(query_text, documents)

    # ---- 1b. Hybrid retrieval: cosine similarity (semantic-ish) ----
    query_vector = _extract_query_term_vector(query_text)
    cosine_scores = {
        f"{s['name']}::{s['file']}": _cosine_similarity(query_vector, s["term_vector"])
        for s in symbols
    }

    # ---- 2. Personalized PageRank, seeded on symbols the task actually names ----
    all_names = {s["name"] for s in symbols}
    seed_symbols = _identify_seed_symbols(query_text, all_names)
    pagerank_scores = personalized_pagerank(conn, seed_symbols) if seed_symbols else {}
    max_pagerank = max(pagerank_scores.values()) if pagerank_scores else 1.0

    # ---- Fuse all three signals into one relevance score per symbol ----
    candidates: List[ContextEntry] = []
    for symbol in symbols:
        key = f"{symbol['name']}::{symbol['file']}"
        bm25_component = bm25_scores.get(key, 0.0)
        cosine_component = cosine_scores.get(key, 0.0)
        pagerank_component = pagerank_scores.get(symbol["name"], 0.0) / max_pagerank if max_pagerank else 0.0

        fused = (
            FUSION_WEIGHTS["bm25"] * bm25_component
            + FUSION_WEIGHTS["cosine"] * cosine_component
            + FUSION_WEIGHTS["pagerank"] * pagerank_component
        )
        if fused <= 0:
            continue

        entry_text = f"- {symbol['file']}: {symbol['signature']}  # {symbol['docstring'][:80]}"
        candidates.append(ContextEntry(
            name=symbol["name"], file=symbol["file"], signature=symbol["signature"],
            docstring=symbol["docstring"], fused_score=fused,
            token_cost=estimate_tokens(entry_text),
        ))

    candidates.sort(key=lambda c: c.fused_score, reverse=True)
    candidates = candidates[:top_k_candidates]

    # ---- 3. Knapsack selection: greedy by relevance-per-token, under budget ----
    candidates.sort(key=lambda c: c.fused_score / max(c.token_cost, 1), reverse=True)

    selected_lines = []
    used_tokens = 0
    for candidate in candidates:
        if used_tokens + candidate.token_cost > budget_tokens:
            continue
        selected_lines.append(
            f"- {candidate.file}: {candidate.signature}  "
            f"# {candidate.docstring[:80]}  (relevance={candidate.fused_score:.2f})"
        )
        used_tokens += candidate.token_cost

    if not selected_lines:
        return "(no relevant existing symbols found within the token budget)"
    return "\n".join(selected_lines)
