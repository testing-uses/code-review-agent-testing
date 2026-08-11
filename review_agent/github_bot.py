"""
review_agent/github_bot.py

The ONLY Git-provider-specific module. Everything upstream of this
(context building, reviewing, deciding) is provider-agnostic and works
directly on diffs and file contents, so migrating this POC from GitHub
to Bitbucket Pipelines later means rewriting just this file
(swap PyGithub calls for Bitbucket REST API / Pipelines API calls).
"""

import os
from typing import Any, Dict

from github import Github

AUTO_APPROVE_LABEL = "agent-approved"
HUMAN_REVIEW_LABEL = "needs-human-review"
REJECTED_LABEL = "agent-rejected"


def render_comment(decision, review_result: Dict[str, Any]) -> str:
    lines = [
        f"## Automated Code Review — **{decision.action}**",
        "",
        f"**Weighted rubric score:** {decision.weighted_score}/100",
        "",
        review_result.get("overall_summary", ""),
        "",
        "### Category scores",
    ]
    for name, score in review_result["category_scores"].items():
        lines.append(f"- {name}: {score}/100")

    findings = review_result["verified_findings"]
    if findings:
        lines.append("\n### Verified findings")
        for f in findings:
            lines.append(
                f"- **[{f['severity']}] {f['title']}** (`{f['file']}:{f.get('line', '?')}`)\n"
                f"  - {f['explanation']}\n"
                f"  - Recommendation: {f['recommendation']}\n"
                f"  - Verification: {f.get('verification_status', 'n/a')} — {f.get('verification_note', '')}"
            )
    else:
        lines.append("\nNo verified findings.")

    lines.append("\n### Decision reasons")
    for reason in decision.reasons:
        lines.append(f"- {reason}")

    lines.append(
        "\n---\n*Generated automatically by the custom LLM review agent. "
        "This is a POC pipeline; treat REJECT/HUMAN_REVIEW as advisory until validated.*"
    )
    return "\n".join(lines)


def apply_decision(repo_full_name: str, pr_number: int, decision, review_result: Dict[str, Any]) -> None:
    gh = Github(os.environ["GITHUB_TOKEN"])
    repo = gh.get_repo(repo_full_name)
    pr = repo.get_pull(pr_number)

    comment_body = render_comment(decision, review_result)
    pr.create_issue_comment(comment_body)

    if decision.action == "AUTO_APPROVE":
        print("Decision is AUTO_APPROVE.")
        print("Creating approval review...")

        pr.create_review(
            event="APPROVE",
            body="Auto-approved by review agent.",
        )

        pr.add_to_labels(AUTO_APPROVE_LABEL)

        print("Attempting to merge pull request...")

        merge_result = pr.merge(
            merge_method="squash",
        )

        print(f"Merge result: merged={merge_result.merged}")
        print(f"Merge message: {merge_result.message}")

        if not merge_result.merged:
            raise RuntimeError(
                f"GitHub did not merge the pull request: "
                f"{merge_result.message}"
            )

    elif decision.action == "HUMAN_REVIEW":
        pr.add_to_labels(HUMAN_REVIEW_LABEL)
        # Intentionally does not merge or request changes — a human decides next.

    elif decision.action == "REJECT":
        pr.add_to_labels(REJECTED_LABEL)
        pr.create_review(event="REQUEST_CHANGES", body="Rejected by review agent. See reasons above.")
        pr.edit(state="closed")

    else:
        raise ValueError(f"Unknown decision action: {decision.action}")