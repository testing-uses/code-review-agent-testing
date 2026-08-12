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
import subprocess
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

def enable_github_auto_merge(
    repo_full_name: str,
    pr_number: int,
) -> None:
    """
    Queue GitHub native auto-merge.

    GitHub will merge the PR after required checks and repository rules
    are satisfied.
    """
    environment = os.environ.copy()
    environment["GH_TOKEN"] = os.environ["GITHUB_TOKEN"]

    command = [
        "gh",
        "pr",
        "merge",
        str(pr_number),
        "--repo",
        repo_full_name,
        "--auto",
        "--squash",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    print(f"gh pr merge exit code: {result.returncode}")

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            "Failed to enable GitHub auto-merge: "
            f"{result.stderr or result.stdout}"
        )

def apply_decision(repo_full_name: str, pr_number: int, decision, review_result: Dict[str, Any]) -> None:
    gh = Github(os.environ["GITHUB_TOKEN"])
    repo = gh.get_repo(repo_full_name)
    pr = repo.get_pull(pr_number)

    comment_body = render_comment(decision, review_result)
    pr.create_issue_comment(comment_body)

    if decision.action == "AUTO_APPROVE":
        print("Decision is AUTO_APPROVE.")

        pr.add_to_labels(AUTO_APPROVE_LABEL)

        enable_github_auto_merge(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
        )

    elif decision.action == "HUMAN_REVIEW":
        print("Decision is HUMAN_REVIEW.")

        pr.add_to_labels(HUMAN_REVIEW_LABEL)

    elif decision.action == "REJECT":
        print("Decision is REJECT.")

        pr.add_to_labels(REJECTED_LABEL)

        pr.create_issue_comment(
            "Rejected by review agent. See reasons above."
        )

        pr.edit(state="closed")

    else:
        raise ValueError(
            f"Unknown decision action: {decision.action}"
        )