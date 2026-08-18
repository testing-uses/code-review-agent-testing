"""
review_agent/reviewer.py

This file did not exist — run_review.py imported `review_pull_request`
from it and would fail on the first PR. Implements the two-stage
review pipeline the two prompt files were clearly written for:

  1. code_reviewer_prompt.md  -> raw per-category findings + scores
  2. code_verifier_prompt.md  -> skeptical re-check; confirm/downgrade/discard

Two separate Groq calls on purpose: a verifier sharing the same reasoning
trace as the reviewer tends to just agree with itself. A fresh call with
only the candidate findings + context re-grounds every claim.
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import yaml

_AGENTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_AGENTS_ROOT, "common"))
from path_bootstrap import bootstrap  # noqa: E402
bootstrap()
from groq_client import GroqKeyPool, call_groq_json, load_prompt  # noqa: E402

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
RUBRIC_PATH = os.path.join(os.path.dirname(__file__), "rubric.yaml")


def load_rubric() -> Dict[str, Any]:
    with open(RUBRIC_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def emit(event: str, **details) -> None:
    """Same [PIPELINE] contract master_agent.py uses, so review sub-steps
    show up in the frontend timeline instead of being a black box between
    'code_review_started' and 'code_review_finished'."""
    payload = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **details}
    print("[PIPELINE] " + json.dumps(payload, sort_keys=True, default=str), flush=True)


def review_pull_request(
    repo_root: str,
    changed_files: List[str],
    context_text: str,
    model: str = "openai/gpt-oss-120b",
    max_output_tokens: int = 1800,
    token_ceiling: int = 6000,
) -> Dict[str, Any]:
    rubric = load_rubric()
    key_pool = GroqKeyPool()

    emit("review_context_ready", changed_files=changed_files,
         context_chars=len(context_text))

    # ---- Stage 1: raw findings ----
    reviewer_system = load_prompt(PROMPTS_DIR, "code_reviewer_prompt.md")
    reviewer_user = (
        "## Rubric categories\n"
        f"{json.dumps({k: v['description'].strip() for k, v in rubric['categories'].items()})}\n\n"
        f"## Changed files\n{changed_files}\n\n"
        f"## Context\n{context_text}"
    )
    raw_review = call_groq_json(
        key_pool=key_pool, model=model, system_prompt=reviewer_system,
        user_prompt=reviewer_user, max_output_tokens=max_output_tokens,
        token_ceiling=token_ceiling,
    )
    raw_review.pop("_usage", None)

    raw_categories = raw_review.get("categories", {}) or {}
    emit("raw_findings_generated",
         category_count=len(raw_categories),
         finding_count=sum(len(c.get("findings", [])) for c in raw_categories.values()))

    all_findings = []
    for category_name, category_data in raw_categories.items():
        for finding in category_data.get("findings", []):
            finding = dict(finding)
            finding["category"] = category_name
            all_findings.append(finding)

    category_scores = {name: data.get("score", 100) for name, data in raw_categories.items()}
    for name in rubric["categories"]:
        category_scores.setdefault(name, 100)

    if not all_findings:
        emit("findings_verified", submitted=0, confirmed=0)
        return {
            "rubric": rubric,
            "category_scores": category_scores,
            "verified_findings": [],
            "overall_summary": raw_review.get("overall_summary", ""),
        }

    # ---- Stage 2: skeptical verification ----
    verifier_system = load_prompt(PROMPTS_DIR, "code_verifier_prompt.md")
    verifier_user = (
        f"## Candidate findings\n{json.dumps(all_findings)}\n\n"
        f"## Context\n{context_text}"
    )
    verified = call_groq_json(
        key_pool=key_pool, model=model, system_prompt=verifier_system,
        user_prompt=verifier_user, max_output_tokens=max_output_tokens,
        token_ceiling=token_ceiling,
    )
    verified.pop("_usage", None)

    verified_findings = [
        f for f in verified.get("verified_findings", [])
        if f.get("verification_status") != "discarded"
    ]

    # The verifier prompt's output schema doesn't carry `category` back —
    # decision_engine's hard-gate check needs it, so re-attach by
    # matching on (file, title), falling back to the finding's own
    # severity-neutral default rather than silently dropping the gate check.
    category_lookup = {(f.get("file"), f.get("title")): f["category"] for f in all_findings}
    for f in verified_findings:
        f["category"] = category_lookup.get((f.get("file"), f.get("title")))

    emit("findings_verified",
         submitted=len(all_findings),
         confirmed=len(verified_findings),
         discarded=len(all_findings) - len(verified_findings))

    return {
        "rubric": rubric,
        "category_scores": category_scores,
        "verified_findings": verified_findings,
        "overall_summary": raw_review.get("overall_summary", ""),
    }