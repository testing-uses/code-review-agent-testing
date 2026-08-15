"""
review_agent/github_bot.py  (v2)

CHANGE: AUTO_APPROVE used to call enable_github_auto_merge(), which
queues `gh pr merge --auto` — the agent merging without any human step.
That directly contradicts the epic doc: "Human Approval: Maintains
human control over final approval and prevents autonomous agents from
independently merging pull requests." AUTO_APPROVE now only labels +
comments (fast lane signal for the human), same as before minus the
actual merge. If you deliberately want opt-in auto-merge later, gate it
behind an explicit env var/flag at the call site — don't make it the
decision engine's default behavior.

Everything else (labels, comment rendering) unchanged.
"""

import os
from typing import Any, Dict
import subprocess
from github import Github

AUTO_APPROVE_LABEL = "agent-approved"
HUMAN_REVIEW_LABEL = "needs-human-review"
REJECTED_LABEL = "agent-rejected"

# Explicit opt-in only. Default is False so a fresh clone of this repo
# never silently merges anything — matches "human maintains final approval".
ALLOW_AUTONOMOUS_AUTO_MERGE = os.environ.get("ALLOW_AUTONOMOUS_AUTO_MERGE", "false").lower() == "true"


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

    if decision.action == "AUTO_APPROVE":
        lines.append(
            "\n*The agent found no blocking issues. A human still needs to "
            "click merge — this pipeline does not merge autonomously.*"
        )

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

    Only called when ALLOW_AUTONOMOUS_AUTO_MERGE is explicitly set — see
    module docstring. GitHub will still merge only after required checks
    and repository rules are satisfied.
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

        if ALLOW_AUTONOMOUS_AUTO_MERGE:
            enable_github_auto_merge(
                repo_full_name=repo_full_name,
                pr_number=pr_number,
            )
        else:
            print("ALLOW_AUTONOMOUS_AUTO_MERGE not set — skipping auto-merge, "
                  "labeled for human merge instead.")

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