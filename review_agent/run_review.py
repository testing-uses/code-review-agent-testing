"""
review_agent/run_review.py  (v2)

Wires: repo_map -> context_builder -> reviewer -> decision_engine -> github_bot
"""

import argparse
import json
import subprocess
import sys

from context_builder import build_context, render_context_for_prompt
from decision_engine import decide
from github_bot import apply_decision
from repo_map import RepoMap
from reviewer import review_pull_request

IGNORED_PATH_PREFIXES = (
    "review_agent/",
    ".github/",
    ".review_agent_cache/",
)

IGNORED_EXACT_PATHS = {
    ".gitignore",
}

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--repo-full-name", required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--context-token-budget", type=int, default=3000)
    args = parser.parse_args()

    all_changed_files = get_changed_files(
    args.repo_root,
    args.base_sha,
    args.head_sha,
    )

    changed_files = [
        file_path
        for file_path in all_changed_files
        if is_reviewable_file(file_path)
    ]

    print(f"All changed files: {all_changed_files}")
    print(f"Reviewable files: {changed_files}")

    if not changed_files:
        print("No reviewable application files changed. Skipping review.")
        sys.exit(0)

    repo_map = RepoMap(args.repo_root)
    repo_map.refresh()  # only re-parses files whose git blob SHA changed

    context_pkg = build_context(
        repo_root=args.repo_root,
        changed_files=changed_files,
        base_sha=args.base_sha,
        head_sha=args.head_sha,
        repo_map=repo_map,
        max_tokens=args.context_token_budget,
    )
    context_text = render_context_for_prompt(context_pkg)

    print(f"Context built: ~{context_pkg.estimated_tokens} tokens, "
          f"truncated={context_pkg.truncated}")

    review_result = review_pull_request(
        repo_root=args.repo_root,
        changed_files=changed_files,
        context_text=context_text,
        model=args.model,
    )

    if context_pkg.removed_symbols_with_usages:
        print("BREAKING CHANGE DETECTED: removed symbol still referenced elsewhere.")
        for symbol, source_file, usages in context_pkg.removed_symbols_with_usages:
            print(f"  {symbol} (removed from {source_file}) used in: "
                f"{[u[0] for u in usages]}")

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

    print(json.dumps({
        "action": decision.action,
        "weighted_score": decision.weighted_score,
        "reasons": decision.reasons,
        "severity_counts": decision.findings_by_severity,
    }, indent=2))

    apply_decision(args.repo_full_name, args.pr_number, decision, review_result)

    if decision.action == "REJECT":
        sys.exit(1)


if __name__ == "__main__":
    main()