"""
agents/orchestrator/dcba.py

Dynamic Context Budget Allocation (DCBA).

Splits a fixed total token budget across agents by task complexity, using
a softmax weighting of complexity scores (same formula family used in
adaptive multi-agent attention/budget orchestration research):

    budget_i = TOTAL_BUDGET * exp(alpha_i) / sum(exp(alpha_j))

This guarantees the allocated budgets always sum to the global ceiling —
which is what prevents three concurrent-ish agents from independently
exceeding the shared Groq account-level TPM limit, the exact failure mode
that caused the original 413 error.

An exponential moving average (EMA) of actual historical usage nudges
future allocations toward what agents really consume, not just the
complexity-score estimate.
"""

import math
from dataclasses import dataclass
from typing import Dict

MIN_BUDGET_PER_AGENT = 800
DEFAULT_TOTAL_BUDGET = 9000  # keep comfortably under a 12,000 TPM ceiling


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
    ema_actual_usage: float | None,
    correction_weight: float = 0.5,
) -> int:
    """Blend the complexity-based estimate with historical actual usage.
    If no history exists yet, return the raw estimate unchanged."""
    if ema_actual_usage is None:
        return raw_budget
    corrected = correction_weight * raw_budget + (1 - correction_weight) * ema_actual_usage
    return int(round(corrected))


def update_ema(previous_ema: float | None, latest_actual: int, alpha: float = 0.3) -> float:
    if previous_ema is None:
        return float(latest_actual)
    return alpha * latest_actual + (1 - alpha) * previous_ema
