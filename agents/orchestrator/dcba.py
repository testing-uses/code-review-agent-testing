"""
agents/orchestrator/dcba.py (v3 -- raised budget floor)

CHANGE from v2: MIN_BUDGET_PER_AGENT and DEFAULT_TOTAL_BUDGET were sized
for the retired llama-3.3-70b-versatile-era prompts. The current
dev_agent_prompt.md system prompt alone is ~2,325 tokens, and a ground-
truth file block can be up to ~1,800 tokens on top of that -- a ~5,000
token ceiling can never fit system prompt + ground truth + a reasonable
output allowance simultaneously, regardless of how well context is
trimmed. Raised both constants so a single-file, ground-truth-driven
edit fits with room to spare. Verify DEFAULT_TOTAL_BUDGET against your
actual Groq plan's TPM (tokens-per-minute) limit before deploying --
this value assumes a plan with at least ~24,000 TPM headroom.

Everything else (two-phase allocation, EMA correction, phase-2
reallocation using real diff signals) is unchanged from v2.
"""

import math
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

MIN_BUDGET_PER_AGENT = 2000
DEFAULT_TOTAL_BUDGET = 11000  # verify against your Groq plan's actual TPM limit

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
    temperature: float = 10.0,
) -> Dict[str, int]:
    """Numerically stable, temperature-scaled Softmax split of total_budget across agents,
    then clamp each to a minimum floor and re-normalize so the sum still equals total_budget."""
    if not complexity_scores:
        return {}

    # Subtract max for numerical stability (prevents overflow for large scores)
    max_score = max(complexity_scores.values())
    temp = max(temperature, 0.1)

    exp_scores = {
        name: math.exp((score - max_score) / temp)
        for name, score in complexity_scores.items()
    }
    total_exp = sum(exp_scores.values()) or 1.0

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
    """Total added+removed lines across the reviewable diff -- the real
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
    """Phase 2. Call this AFTER the PR is created -- real diff, real
    complexity, real remaining budget.

    Caps the review agent at whatever's actually left of the shared
    ceiling after what the Dev Agent actually used (not its allocation --
    agents routinely use less than they're given)."""
    signals = ComplexitySignals(
        files_changed=len(changed_files),
        lines_changed=diff_line_count(repo_root, base_sha, head_sha, changed_files),
        is_security_sensitive=looks_security_sensitive(changed_files),
        jira_priority_weight=1.0,
    )
    score = complexity_score(signals)

    remaining_budget = max(total_budget - dev_actual_tokens_used, min_budget)

    scale = min(score / 40.0, 1.0)
    raw_budget = max(int(remaining_budget * scale), min_budget)
    raw_budget = min(raw_budget, remaining_budget)

    return apply_ema_correction(raw_budget, review_ema)