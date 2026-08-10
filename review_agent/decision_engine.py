"""
review_agent/decision_engine.py

Turns rubric scores + verified findings into one of three actions:
  AUTO_APPROVE  -> push/merge directly
  HUMAN_REVIEW  -> pause, flag a human reviewer
  REJECT        -> close the PR with a reason

Hard gates are evaluated BEFORE the weighted score, and always win.
This prevents a high aggregate score from masking one severe issue.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


@dataclass
class Decision:
    action: str  # AUTO_APPROVE | HUMAN_REVIEW | REJECT
    weighted_score: float
    reasons: List[str] = field(default_factory=list)
    findings_by_severity: Dict[str, int] = field(default_factory=dict)


def compute_weighted_score(rubric: Dict[str, Any], category_scores: Dict[str, float]) -> float:
    total = 0.0
    weight_sum = 0.0
    for name, cfg in rubric["categories"].items():
        score = category_scores.get(name, 0)
        weight = cfg["weight"]
        total += score * weight
        weight_sum += weight
    return round(total / weight_sum, 2) if weight_sum else 0.0


def check_hard_gates(rubric: Dict[str, Any], findings: List[Dict[str, Any]]) -> List[str]:
    """Return a list of gate-violation reasons; empty list = no hard gate triggered
    at REJECT level (backward_compatibility gate is checked separately for
    HUMAN_REVIEW-forcing behavior)."""
    reject_reasons = []
    for name, cfg in rubric["categories"].items():
        if not cfg.get("hard_gate"):
            continue
        gate_severity = cfg["hard_gate_severity"]
        for finding in findings:
            if finding.get("category") != name:
                continue
            if SEVERITY_RANK.get(finding.get("severity", "LOW"), 0) >= SEVERITY_RANK[gate_severity]:
                if gate_severity == "CRITICAL":
                    reject_reasons.append(
                        f"Hard gate triggered: {name} finding '{finding['title']}' "
                        f"is {finding['severity']} (gate={gate_severity})"
                    )
    return reject_reasons


def check_forced_human_review(rubric: Dict[str, Any], findings: List[Dict[str, Any]]) -> List[str]:
    reasons = []
    for name, cfg in rubric["categories"].items():
        if not cfg.get("hard_gate") or cfg.get("hard_gate_severity") == "CRITICAL":
            continue
        gate_severity = cfg["hard_gate_severity"]
        for finding in findings:
            if finding.get("category") != name:
                continue
            if SEVERITY_RANK.get(finding.get("severity", "LOW"), 0) >= SEVERITY_RANK[gate_severity]:
                reasons.append(
                    f"Forced human review: {name} finding '{finding['title']}' "
                    f"is {finding['severity']} (gate={gate_severity})"
                )
    return reasons


def decide(review_result: Dict[str, Any]) -> Decision:
    rubric = review_result["rubric"]
    findings = review_result["verified_findings"]
    category_scores = review_result["category_scores"]
    thresholds = rubric["decision_thresholds"]

    weighted_score = compute_weighted_score(rubric, category_scores)

    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for f in findings:
        sev = f.get("severity", "LOW")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # 1. Hard reject gates
    reject_reasons = check_hard_gates(rubric, findings)
    if reject_reasons:
        return Decision("REJECT", weighted_score, reject_reasons, severity_counts)

    # 2. Forced human-review gates (e.g. breaking-change risk)
    human_reasons = check_forced_human_review(rubric, findings)
    if human_reasons:
        return Decision("HUMAN_REVIEW", weighted_score, human_reasons, severity_counts)

    # 3. Any unresolved CRITICAL finding anywhere -> reject as a safety net
    if severity_counts["CRITICAL"] > 0:
        return Decision(
            "REJECT",
            weighted_score,
            [f"{severity_counts['CRITICAL']} unresolved CRITICAL finding(s) present"],
            severity_counts,
        )

    # 4. Any HIGH finding -> at least human review, regardless of score
    if severity_counts["HIGH"] > 0:
        return Decision(
            "HUMAN_REVIEW",
            weighted_score,
            [f"{severity_counts['HIGH']} unresolved HIGH finding(s) present"],
            severity_counts,
        )

    # 5. Weighted score thresholds
    if weighted_score >= thresholds["auto_approve_min_score"]:
        return Decision("AUTO_APPROVE", weighted_score, ["Score meets auto-approve threshold"], severity_counts)
    if weighted_score >= thresholds["human_review_min_score"]:
        return Decision("HUMAN_REVIEW", weighted_score, ["Score in moderate range"], severity_counts)

    return Decision("REJECT", weighted_score, ["Score below minimum acceptable threshold"], severity_counts)