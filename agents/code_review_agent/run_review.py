"""
review_agent/run_review.py  (v3)

Wires: central KB (kb.sqlite3) -> context_builder -> reviewer -> decision_engine -> github_bot

CHANGES from v2:
  - Exposes run_review_for_pr(), a plain function master_agent.py already
    tried to import (`from run_review import run_review_for_pr`) — that
    function never existed, so every orchestrated run silently skipped
    real code review and reported "NOT_WIRED_YET". Fixed here.
  - No longer builds/depends on repo_map.py's separate symbol cache —
    context_builder now resolves symbols from the same kb.sqlite3 the
    Dev Agent uses, so pass --db-path / db_path through instead.
  - Emits [PIPELINE]-prefixed progress events (same contract as
    master_agent.emit / reviewer.emit) so the frontend timeline can show
    real review sub-steps instead of a black box.
  - Returns a result dict keyed by "action" using decision_engine's real
    vocabulary (AUTO_APPROVE / HUMAN_REVIEW / REJECT), not an invented
    "PASS"/"status" string master_agent.py was comparing against and
    could never match.
"""

import argparse
import json
import subprocess
import sys

import os
import sys
from datetime import datetime, timezone

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(AGENT_DIR, "..", "knowledge_base")
AGENTS_ROOT = os.path.dirname(AGENT_DIR)

sys.path.insert(0, os.path.join(AGENTS_ROOT, "common"))
from path_bootstrap import bootstrap  # noqa: E402
bootstrap()

from context_builder import build_context, render_context_for_prompt
from decision_engine import decide
from github_bot import apply_decision
from reviewer import review_pull_request


IGNORED_PATH_PREFIXES = (
    "review_agent/",
    "agents/",
    ".github/",
    ".review_agent_cache/",
)

IGNORED_EXACT_PATHS = {
    ".gitignore",
}


def emit(event: str, **details) -> None:
    payload = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **details}
    print("[PIPELINE] " + json.dumps(payload, sort_keys=True, default=str), flush=True)


def is_reviewable_file(file_path: str) -> bool:
    normalized_path = file_path.replace("\\", "/")

    if normalized_path in IGNORED_EXACT_PATHS:
        return False

    return not normalized_path.startswith(IGNORED_PATH_PREFIXES)


def get_changed_files(repo_root: str, base_sha: str, head_sha: str):
    result = subprocess.run(
        ["git", "diff", "--name-only", base_sha, head_sha],
        cwd=repo_root, capture_output=True, text=True, check=True,
    )
    return [f for f in result.stdout.strip().splitlines() if f]


def run_review_for_pr(
    repo_root: str,
    base_sha: str,
    head_sha: str,
    repo_full_name: str,
    pr_number: int,
    budget_tokens: int = 3000,
    db_path: str = None,
    model: str = "openai/gpt-oss-120b",
    post_to_github: bool = True,
) -> dict:
    """The function master_agent.py has been trying to import all along.
    Returns {"action": ..., "weighted_score": ..., "reasons": [...],
    "severity_counts": {...}} — "action" is one of AUTO_APPROVE /
    HUMAN_REVIEW / REJECT / ERROR, matching decision_engine's real
    vocabulary so callers don't have to guess at a "PASS" string that
    never gets emitted."""
    if db_path is None:
        db_path = os.path.join(KB_DIR, "kb.sqlite3")

    all_changed_files = get_changed_files(repo_root, base_sha, head_sha)
    changed_files = [f for f in all_changed_files if is_reviewable_file(f)]

    emit("review_files_identified", all_changed=all_changed_files, reviewable=changed_files)

    if not changed_files:
        emit("review_skipped", reason="no reviewable application files changed")
        return {"action": "AUTO_APPROVE", "weighted_score": 100.0,
                "reasons": ["No reviewable application files changed"], "severity_counts": {}}

    context_pkg = build_context(
        repo_root=repo_root,
        changed_files=changed_files,
        base_sha=base_sha,
        head_sha=head_sha,
        db_path=db_path,
        max_tokens=budget_tokens,
    )
    context_text = render_context_for_prompt(context_pkg)

    emit("review_context_built",
         estimated_tokens=context_pkg.estimated_tokens,
         truncated=context_pkg.truncated)

    review_result = review_pull_request(
        repo_root=repo_root,
        changed_files=changed_files,
        context_text=context_text,
        model=model,
    )

    if context_pkg.removed_symbols_with_usages:
        emit("breaking_change_detected",
             removed_symbols=[s for s, _, _ in context_pkg.removed_symbols_with_usages])
        decision = decide(review_result)
        if decision.action == "AUTO_APPROVE":
            decision.action = "HUMAN_REVIEW"
            decision.reasons.insert(
                0,
                "Overridden: a removed symbol is still referenced elsewhere in "
                "the codebase (likely breaking change). Deterministic check, "
                "not LLM-dependent.",
            )
    else:
        decision = decide(review_result)

    emit("review_decision_made",
         action=decision.action,
         weighted_score=decision.weighted_score,
         reasons=decision.reasons)

    if post_to_github:
        apply_decision(repo_full_name, pr_number, decision, review_result)
        emit("review_posted_to_github", pr_number=pr_number, action=decision.action)

    return {
        "action": decision.action,
        "weighted_score": decision.weighted_score,
        "reasons": decision.reasons,
        "severity_counts": decision.findings_by_severity,
        "usage": review_result.get("usage", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--repo-full-name", required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--context-token-budget", type=int, default=3000)
    parser.add_argument("--db-path", default=os.path.join(KB_DIR, "kb.sqlite3"))
    args = parser.parse_args()

    result = run_review_for_pr(
        repo_root=args.repo_root,
        base_sha=args.base_sha,
        head_sha=args.head_sha,
        repo_full_name=args.repo_full_name,
        pr_number=args.pr_number,
        budget_tokens=args.context_token_budget,
        db_path=args.db_path,
        model=args.model,
        post_to_github=True,
    )

    print(json.dumps(result, indent=2))

    if result["action"] == "REJECT":
        sys.exit(1)


if __name__ == "__main__":
    main()