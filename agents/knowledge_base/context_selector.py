"""
agents/knowledge_base/context_selector.py  (v2)

The unified context-assembly service used by BOTH the Dev Agent and the
Code Review Agent. Implements:

    1. Hybrid retrieval  = BM25 (lexical) + cosine similarity (semantic).
    2. Personalized PageRank = ranks symbols by graph proximity to the
       task's seed files/symbols (same algorithm family as Aider's repo map).
    3. Budget-constrained MMR selection (NEW — see below).

CHANGE from v1: step 3 used to be plain greedy selection by
fused_relevance/token_cost. That's a fine knapsack heuristic for
maximizing *relevance density*, but it has a known failure mode: if
several top-ranked symbols are near-duplicates of each other (e.g. three
overloaded variants of the same helper, or a function and the thin
wrapper that just calls it), greedy selection happily spends budget on
all three, because each one individually still scores well — even
though the second and third add almost no new information over the
first.

Replaced with Maximal Marginal Relevance (Carbonell & Goldstein, 1998):
at each step, pick the candidate that maximizes
    lambda * relevance - (1 - lambda) * max_similarity_to_already_selected
instead of just picking by relevance. This is a direct, well-understood
fix for exactly the redundancy problem a fixed token budget makes worse
— you want the budget spent on *diverse* relevant context, not five
near-identical entries about the same thing. Diversity is measured with
the same cosine similarity over term vectors already computed for
retrieval, so this adds no new infrastructure.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Set

_AGENTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_AGENTS_ROOT, "common"))
from path_bootstrap import bootstrap  # noqa: E402
bootstrap()

from bm25 import compute_bm25_scores  # noqa: E402
from kb_query import _cosine_similarity, personalized_pagerank  # noqa: E402
from groq_client import estimate_tokens  # noqa: E402

FUSION_WEIGHTS = {"bm25": 0.4, "cosine": 0.35, "pagerank": 0.25}
MMR_LAMBDA = 0.7  # 0.7 relevance / 0.3 diversity — tune down for more diverse, less relevant


@dataclass
class ContextEntry:
    name: str
    file: str
    signature: str
    docstring: str
    fused_score: float
    token_cost: int
    term_vector: Dict[str, int] = field(default_factory=dict)


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


def _mmr_select(candidates: List[ContextEntry], budget_tokens: int, mmr_lambda: float = MMR_LAMBDA) -> List[ContextEntry]:
    """Budget-constrained MMR: repeatedly pick the candidate that best
    trades off relevance against redundancy with what's already selected,
    normalized by token cost. Eliminates negative division anomalies."""
    selected: List[ContextEntry] = []
    remaining = list(candidates)
    used_tokens = 0

    while remaining:
        affordable = [c for c in remaining if used_tokens + c.token_cost <= budget_tokens]
        if not affordable:
            break

        best_entry = None
        best_efficiency = float("-inf")
        for candidate in affordable:
            redundancy = 0.0
            if selected:
                redundancy = max(
                    _cosine_similarity(candidate.term_vector, s.term_vector) for s in selected
                )
            marginal_utility = mmr_lambda * candidate.fused_score - (1 - mmr_lambda) * redundancy
            
            # Density ratio: positive utility is boosted by low token cost;
            # negative utility (pure redundancy) is heavily discounted
            if marginal_utility > 0:
                efficiency = marginal_utility / max(candidate.token_cost, 1)
            else:
                efficiency = marginal_utility * max(candidate.token_cost, 1)

            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_entry = candidate

        if not best_entry or (selected and best_efficiency < 0):
            # If remaining affordable candidates only add pure redundancy, stop to conserve budget
            break

        selected.append(best_entry)
        used_tokens += best_entry.token_cost
        remaining.remove(best_entry)

    return selected


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
            term_vector=symbol["term_vector"],
        ))

    candidates.sort(key=lambda c: c.fused_score, reverse=True)
    candidates = candidates[:top_k_candidates]

    # ---- 3. Budget-constrained MMR selection (relevance vs. redundancy) ----
    selected = _mmr_select(candidates, budget_tokens)

    if not selected:
        return "(no relevant existing symbols found within the token budget)"

    return "\n".join(
        f"- {c.file}: {c.signature}  # {c.docstring[:80]}  (relevance={c.fused_score:.2f})"
        for c in selected
    )