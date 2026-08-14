"""
agents/orchestrator/master_agent.py  (v2 — full pipeline, runs inside
GitHub Actions since the org network blocks direct Groq access locally)

Flow:
    1. DCBA allocates token budgets for Dev Agent and Code Review Agent,
       informed by task size AND how much of the knowledge base the task
       actually touches (bigger KB footprint -> higher complexity score).
    2. Dev Agent produces diffs/new files (no git calls).
    3. Orchestrator applies git operations: branch, commit, push, open PR.
    4. Code Review Agent runs in the SAME job against the new PR, using
       its own DCBA budget.
    5. Deterministic state machine decides the outcome. The pipeline NEVER
       auto-merges — it stops at HUMAN_APPROVAL_REQUIRED or WORKFLOW_FAILED.
"""

import argparse
import json
import os
import sys
import time
import uuid
from typing import Any, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dev_agent"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "knowledge_base"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code_review_agent"))

from groq_client import GroqKeyPool, call_groq_json, load_prompt  # noqa: E402
from github_ops import commit_and_push, create_pull_request, get_base_sha  # noqa: E402
from dcba import (  # noqa: E402
    ComplexitySignals, complexity_score, allocate_budgets,
    apply_ema_correction, update_ema, DEFAULT_TOTAL_BUDGET,
)
from metrics_store import record_metric, build_metric, get_last_ema  # noqa: E402
from state_machine import WorkflowState, transition, InvalidTransitionError  # noqa: E402
from kb_schema import get_connection  # noqa: E402
from context_selector import select_context  # noqa: E402

import dev_agent  # noqa: E402

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.jsonl")
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def estimate_task_complexity(db_path: str, task_text: str) -> ComplexitySignals:
    """Bigger KB footprint (more matching symbols) -> more complex task ->
    bigger budget. This is the "orchestrator evaluates how big the task is
    using the knowledge base" step you described."""
    kb_footprint = 0
    if os.path.exists(db_path):
        conn = get_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM symbols")
        total_symbols = cursor.fetchone()[0] or 1
        matched = select_context(conn, task_text, budget_tokens=999999)
        kb_footprint = matched.count("\n- ") if matched else 0
        conn.close()

    return ComplexitySignals(
        task_text_length=len(task_text),
        files_changed=max(1, kb_footprint // 3),
        jira_priority_weight=1.0,
    )


def run_pipeline(
    task_text: str,
    repo_root: str,
    db_path: str,
    repo_full_name: str,
    base_branch: str = "main",
    total_budget: int = DEFAULT_TOTAL_BUDGET,
) -> Dict[str, Any]:
    task_id = str(uuid.uuid4())[:8]
    state = WorkflowState.TASK_RECEIVED
    report: Dict[str, Any] = {"task_id": task_id}

    # ---- Step 1: DCBA, informed by the knowledge base ----
    dev_ema = get_last_ema(METRICS_PATH, "dev_agent")
    review_ema = get_last_ema(METRICS_PATH, "code_review_agent")

    dev_signals = estimate_task_complexity(db_path, task_text)
    review_signals = ComplexitySignals(files_changed=1, lines_changed=20, jira_priority_weight=1.0)

    raw_scores = {
        "dev_agent": complexity_score(dev_signals),
        "code_review_agent": complexity_score(review_signals),
    }
    raw_budgets = allocate_budgets(raw_scores, total_budget=total_budget)
    budgets = {
        "dev_agent": apply_ema_correction(raw_budgets["dev_agent"], dev_ema),
        "code_review_agent": apply_ema_correction(raw_budgets["code_review_agent"], review_ema),
    }
    report["dcba_budgets"] = budgets

    # ---- Step 2: Dev Agent (content only, no git) ----
    state = transition(state, WorkflowState.DEV_IN_PROGRESS)
    dev_start = time.time()
    dev_result = dev_agent.run(
        task_text=task_text, repo_root=repo_root, db_path=db_path,
        allocated_budget_tokens=budgets["dev_agent"],
    )
    dev_latency = (time.time() - dev_start) * 1000

    usage = dev_result.get("usage", {})
    new_dev_ema = update_ema(dev_ema, usage.get("total_tokens", 0) or 0)
    record_metric(METRICS_PATH, build_metric(
        agent="dev_agent", task_id=task_id, allocated_budget_tokens=budgets["dev_agent"],
        actual_prompt_tokens=usage.get("prompt_tokens"), actual_completion_tokens=usage.get("completion_tokens"),
        latency_ms=dev_latency, result_status=dev_result["status"],
        groq_key_used=usage.get("key_index"), ema_total_tokens=new_dev_ema,
    ))
    report["dev_agent"] = dev_result

    if dev_result["status"] != "FILES_READY":
        state = transition(state, WorkflowState.DEV_BLOCKED)
        state = transition(state, WorkflowState.WORKFLOW_FAILED)
        report["final_state"] = state.value
        return report

    # ---- Step 3: Orchestrator owns git — branch, commit, push, open PR ----
    branch_name = f"dev-agent-{task_id}"
    jira_key = dev_result.get("jira_key") or "DEV"
    commit_message = f"{jira_key}: {dev_result.get('summary', 'automated change')}"

    try:
        head_sha = commit_and_push(
            repo_root=repo_root, branch_name=branch_name,
            changed_files=dev_result["changed_files"], commit_message=commit_message,
        )
        base_sha = get_base_sha(repo_root, base_branch)
        pr_number, head_sha = create_pull_request(
            repo_full_name=repo_full_name, branch_name=branch_name, base_branch=base_branch,
            title=f"[{jira_key}] {dev_result.get('summary', 'Automated change')}",
            body=f"Automated implementation by Dev Agent.\n\nTask:\n{task_text}",
        )
    except Exception as error:
        state = transition(state, WorkflowState.DEV_BLOCKED)
        state = transition(state, WorkflowState.WORKFLOW_FAILED)
        report["final_state"] = state.value
        report["git_error"] = str(error)
        return report

    state = transition(state, WorkflowState.PR_CREATED)
    report["pull_request"] = {"number": pr_number, "branch": branch_name, "head_sha": head_sha}

    # ---- Step 4: Code Review Agent, same job, its own DCBA budget ----
    # Integration point: agents/code_review_agent/run_review.py, relocated
    # and updated to accept (repo_root, base_sha, head_sha, repo_full_name,
    # pr_number, budget_tokens) and to load its prompts from
    # agents/prompts/code_reviewer_prompt.md / code_verifier_prompt.md.
    state = transition(state, WorkflowState.CODE_REVIEW_IN_PROGRESS)
    try:
        from run_review import run_review_for_pr  # noqa: E402
        review_result = run_review_for_pr(
            repo_root=repo_root, base_sha=base_sha, head_sha=head_sha,
            repo_full_name=repo_full_name, pr_number=pr_number,
            budget_tokens=budgets["code_review_agent"],
        )
    except ImportError:
        review_result = {"status": "NOT_WIRED_YET"}

    report["code_review_agent"] = review_result
    decision_action = review_result.get("status")

    if decision_action == "REJECT":
        state = transition(state, WorkflowState.CODE_REVIEW_BLOCKED)
        state = transition(state, WorkflowState.WORKFLOW_FAILED)
    else:
        # PASS, HUMAN_REVIEW, or NOT_WIRED_YET all stop for a human — never auto-merge.
        target = (
            WorkflowState.CODE_REVIEW_PASSED
            if decision_action == "PASS" else WorkflowState.CODE_REVIEW_BLOCKED
        )
        state = transition(state, target)
        state = transition(state, WorkflowState.HUMAN_APPROVAL_REQUIRED)

    report["final_state"] = state.value

    # ---- Step 5: LLM-generated human-facing summary (explanation only) ----
    try:
        summary_prompt = load_prompt(PROMPTS_DIR, "orchestrator_summary_prompt.md")
        key_pool = GroqKeyPool()
        summary_user_prompt = (
            f"## Task\n{task_text}\n\n## Dev Agent result\n{json.dumps(dev_result)}\n\n"
            f"## Code Review Agent result\n{json.dumps(review_result)}\n\n"
            f"## Final workflow state\n{state.value}"
        )
        summary = call_groq_json(
            key_pool=key_pool, model=DEFAULT_MODEL, system_prompt=summary_prompt,
            user_prompt=summary_user_prompt, max_output_tokens=400, token_ceiling=2000,
        )
        summary.pop("_usage", None)
        report["human_readable_summary"] = summary
    except Exception as error:
        report["human_readable_summary"] = {"summary": f"(summary generation failed: {error})"}

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-text", help="Task description text (or use --task-file)")
    parser.add_argument("--task-file", help="Path to a file containing the task description")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--db-path", default="agents/knowledge_base/kb.sqlite3")
    parser.add_argument("--repo-full-name", required=True)
    parser.add_argument("--base-branch", default="main")
    parser.add_argument("--total-budget", type=int, default=DEFAULT_TOTAL_BUDGET)
    args = parser.parse_args()

    if args.task_file:
        with open(args.task_file, "r", encoding="utf-8") as fh:
            task_text = fh.read().strip()
    elif args.task_text:
        task_text = args.task_text.strip()
    else:
        raise SystemExit("Provide either --task-text or --task-file")

    try:
        report = run_pipeline(
            task_text=task_text, repo_root=args.repo_root, db_path=args.db_path,
            repo_full_name=args.repo_full_name, base_branch=args.base_branch,
            total_budget=args.total_budget,
        )
    except InvalidTransitionError as error:
        report = {"error": str(error)}

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
