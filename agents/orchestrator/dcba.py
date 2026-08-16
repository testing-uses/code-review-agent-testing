"""
agents/orchestrator/dcba.py  (v2 — two-phase allocation)

Dynamic Context Budget Allocation (DCBA).

Phase 1 (unchanged): softmax-weighted split of the total budget across
agents by complexity score, computed BEFORE the Dev Agent has run. This
is necessarily a guess for the review agent's side — you don't know the
diff size or whether it touches anything sensitive before it exists.

Phase 2 (NEW — reallocate_review_budget): once the Dev Agent has actually
produced a diff, replace that guess with real signals — actual files/
lines changed, whether the diff touches anything that looks security-
sensitive — and cap the review agent's budget by what's ACTUALLY left of
the shared ceiling after the Dev Agent's real usage, not its original
estimate. This is what actually protects the shared Groq TPM limit;
phase 1 alone only protects against two budgets that were both guesses.
"""

import math
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

MIN_BUDGET_PER_AGENT = 800
DEFAULT_TOTAL_BUDGET = 9000  # keep comfortably under a 12,000 TPM ceiling

SECURITY_SENSITIVE_HINTS = (
    "auth", "secret", "password", "token", "crypto", "permission",
    "security", "acl", "session", "credential", "login",
)


@dataclass
class ComplexitySignals:
    files_changed: int = 0
    lines_changed: int = 0
    is_security_sensitive: bool = False
    jira_priority_weight: float = 1.0
    previous_retry_count: int = 0
    task_text_length: int = 0  # used pre-diff, for the Dev Agent's first pass


def complexity_score(signals: ComplexitySignals) -> float:
    score = (
        signals.files_changed * 1.5
        + signals.lines_changed * 0.03
        + (15.0 if signals.is_security_sensitive else 0.0)
        + signals.jira_priority_weight * 5.0
        + signals.previous_retry_count * 8.0
        + signals.task_text_length * 0.01
    )
    return max(score, 0.1)  # avoid exp(0) collapsing all weights equally


def allocate_budgets(
    complexity_scores: Dict[str, float],
    total_budget: int = DEFAULT_TOTAL_BUDGET,
    min_budget: int = MIN_BUDGET_PER_AGENT,
) -> Dict[str, int]:
    """Softmax-weighted split of total_budget across agents, then clamp each
    to a minimum floor and re-normalize so the sum still equals total_budget."""
    if not complexity_scores:
        return {}

    exp_scores = {name: math.exp(score) for name, score in complexity_scores.items()}
    total_exp = sum(exp_scores.values())

    raw_budgets = {
        name: (exp_score / total_exp) * total_budget
        for name, exp_score in exp_scores.items()
    }

    clamped = {name: max(budget, min_budget) for name, budget in raw_budgets.items()}
    clamped_total = sum(clamped.values())

    if clamped_total > total_budget:
        scale = total_budget / clamped_total
        clamped = {name: budget * scale for name, budget in clamped.items()}

    return {name: int(round(budget)) for name, budget in clamped.items()}


def apply_ema_correction(
    raw_budget: int,
    ema_actual_usage: Optional[float],
    correction_weight: float = 0.5,
) -> int:
    """Blend the complexity-based estimate with historical actual usage.
    If no history exists yet, return the raw estimate unchanged."""
    if ema_actual_usage is None:
        return raw_budget
    corrected = correction_weight * raw_budget + (1 - correction_weight) * ema_actual_usage
    return int(round(corrected))


def update_ema(previous_ema: Optional[float], latest_actual: int, alpha: float = 0.3) -> float:
    if previous_ema is None:
        return float(latest_actual)
    return alpha * latest_actual + (1 - alpha) * previous_ema


def diff_line_count(repo_root: str, base_sha: str, head_sha: str, changed_files: List[str]) -> int:
    """Total added+removed lines across the reviewable diff — the real
    'lines_changed' signal, vs. the hardcoded 20 the review agent's budget
    used to be computed from."""
    if not changed_files:
        return 0
    result = subprocess.run(
        ["git", "diff", "--numstat", base_sha, head_sha, "--", *changed_files],
        cwd=repo_root, capture_output=True, text=True,
    )
    total = 0
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        added, removed = parts[0], parts[1]
        total += (int(added) if added.isdigit() else 0) + (int(removed) if removed.isdigit() else 0)
    return total


def looks_security_sensitive(changed_files: List[str]) -> bool:
    return any(hint in f.lower() for f in changed_files for hint in SECURITY_SENSITIVE_HINTS)


def reallocate_review_budget(
    repo_root: str,
    base_sha: str,
    head_sha: str,
    changed_files: List[str],
    dev_actual_tokens_used: int,
    total_budget: int,
    review_ema: Optional[float],
    min_budget: int = MIN_BUDGET_PER_AGENT,
) -> int:
    """Phase 2. Call this AFTER the PR is created — real diff, real
    complexity, real remaining budget.

    Caps the review agent at whatever's actually left of the shared
    ceiling after what the Dev Agent actually used (not its allocation —
    agents routinely use less than they're given), so a big Dev Agent
    call doesn't leave the review agent starved for no reason, and a
    small one doesn't get treated the same as a 40-file security-touching
    diff, which is what the static ComplexitySignals(files_changed=1,
    lines_changed=20) placeholder was doing before."""
    signals = ComplexitySignals(
        files_changed=len(changed_files),
        lines_changed=diff_line_count(repo_root, base_sha, head_sha, changed_files),
        is_security_sensitive=looks_security_sensitive(changed_files),
        jira_priority_weight=1.0,
    )
    score = complexity_score(signals)

    remaining_budget = max(total_budget - dev_actual_tokens_used, min_budget)

    # score=40 already reflects a large, security-touching diff — cap the
    # growth curve there so one huge PR can't claim the entire remaining
    # budget and leave nothing as a floor.
    scale = min(score / 40.0, 1.0)
    raw_budget = max(int(remaining_budget * scale), min_budget)
    raw_budget = min(raw_budget, remaining_budget)

    return apply_ema_correction(raw_budget, review_ema)