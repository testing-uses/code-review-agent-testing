"""agents/orchestrator/master_agent.py  (v3)

CHANGES from v2:
  - Uses path_bootstrap.bootstrap() instead of hand-rolled relative
    sys.path.append() calls — fixes the ImportError class of bug where
    context_selector (and friends) could be reached via two different
    sys.path entries in the same process.
  - Phase-2 DCBA: after the Dev Agent's diff actually exists, the review
    agent's budget is recomputed from real signals (dcba.reallocate_review_budget)
    instead of staying pinned to the static placeholder guessed before
    the diff existed.
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

_AGENTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_AGENTS_ROOT, "common"))
from path_bootstrap import bootstrap  # noqa: E402
bootstrap()

from groq_client import GroqKeyPool, call_groq_json, load_prompt  # noqa: E402
from github_ops import commit_and_push, create_pull_request, get_base_sha  # noqa: E402
from dcba import (  # noqa: E402
    ComplexitySignals,
    complexity_score,
    allocate_budgets,
    apply_ema_correction,
    update_ema,
    reallocate_review_budget,
    DEFAULT_TOTAL_BUDGET,
)
from metrics_store import record_metric, build_metric, get_last_ema  # noqa: E402
from state_machine import WorkflowState, transition, InvalidTransitionError  # noqa: E402
from kb_schema import get_connection  # noqa: E402
from context_selector import select_context  # noqa: E402
import dev_agent  # noqa: E402

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.jsonl")
DEFAULT_MODEL = "openai/gpt-oss-120b"


def emit(event: str, **details) -> dict:
    payload = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    print("[PIPELINE] " + json.dumps(payload, sort_keys=True, default=str), flush=True)
    return payload


def estimate_task_complexity(db_path: str, task_text: str) -> ComplexitySignals:
    kb_footprint = 0
    total_symbols = 0

    if os.path.exists(db_path):
        conn = get_connection(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM symbols")
            total_symbols = cursor.fetchone()[0] or 0
            matched = select_context(conn, task_text, budget_tokens=999999)
            kb_footprint = matched.count("\n- ") if matched else 0
        finally:
            conn.close()

    emit(
        "knowledge_base_context_analyzed",
        path=db_path,
        exists=os.path.exists(db_path),
        total_symbols=total_symbols,
        matched_context_entries=kb_footprint,
    )

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

    emit("task_received", task_id=task_id, task=task_text)
    emit("knowledge_base_checked", path=db_path, exists=os.path.exists(db_path))

    dev_ema = get_last_ema(METRICS_PATH, "dev_agent")
    review_ema = get_last_ema(METRICS_PATH, "code_review_agent")

    dev_signals = estimate_task_complexity(db_path, task_text)
    review_signals = ComplexitySignals(
        files_changed=1,
        lines_changed=20,
        jira_priority_weight=1.0,
    )

    raw_scores = {
        "dev_agent": complexity_score(dev_signals),
        "code_review_agent": complexity_score(review_signals),
    }
    raw_budgets = allocate_budgets(raw_scores, total_budget=total_budget)
    budgets = {
        "dev_agent": apply_ema_correction(raw_budgets["dev_agent"], dev_ema),
        "code_review_agent": apply_ema_correction(
            raw_budgets["code_review_agent"], review_ema
        ),
    }
    report["dcba_budgets_phase1"] = dict(budgets)

    emit(
        "dcba_allocated",
        total_budget=total_budget,
        raw_scores=raw_scores,
        dev_agent_tokens=budgets["dev_agent"],
        code_review_agent_tokens_initial_estimate=budgets["code_review_agent"],
    )

    state = transition(state, WorkflowState.DEV_IN_PROGRESS)
    emit(
        "dev_agent_started",
        allocated_tokens=budgets["dev_agent"],
        context_budget_tokens=int(budgets["dev_agent"] * 0.6),
        db_path=db_path,
    )

    dev_start = time.time()
    dev_result = dev_agent.run(
        task_text=task_text,
        repo_root=repo_root,
        db_path=db_path,
        allocated_budget_tokens=budgets["dev_agent"],
    )
    dev_latency = (time.time() - dev_start) * 1000

    usage = dev_result.get("usage", {})
    dev_actual_tokens = usage.get("total_tokens") or 0
    new_dev_ema = update_ema(dev_ema, dev_actual_tokens)
    record_metric(
        METRICS_PATH,
        build_metric(
            agent="dev_agent",
            task_id=task_id,
            allocated_budget_tokens=budgets["dev_agent"],
            actual_prompt_tokens=usage.get("prompt_tokens"),
            actual_completion_tokens=usage.get("completion_tokens"),
            latency_ms=dev_latency,
            result_status=dev_result["status"],
            groq_key_used=usage.get("key_index"),
            ema_total_tokens=new_dev_ema,
        ),
    )
    report["dev_agent"] = dev_result

    emit(
        "dev_agent_finished",
        status=dev_result.get("status"),
        changed_files=dev_result.get("changed_files", []),
        apply_errors=dev_result.get("apply_errors", []),
        blocked_reason=dev_result.get("blocked_reason", ""),
        usage=usage,
        latency_ms=dev_latency,
    )

    if dev_result["status"] != "FILES_READY":
        state = transition(state, WorkflowState.DEV_BLOCKED)
        state = transition(state, WorkflowState.WORKFLOW_FAILED)
        report["final_state"] = state.value
        emit(
            "workflow_failed",
            task_id=task_id,
            stage="dev_agent",
            reason=dev_result.get("blocked_reason", ""),
        )
        return report

    branch_name = f"dev-agent-{task_id}"
    jira_key = dev_result.get("jira_key") or "DEV"
    commit_message = f"{jira_key}: {dev_result.get('summary', 'automated change')}"

    emit(
        "git_operations_started",
        branch=branch_name,
        changed_files=dev_result["changed_files"],
        commit_message=commit_message,
    )

    try:
        head_sha = commit_and_push(
            repo_root=repo_root,
            branch_name=branch_name,
            changed_files=dev_result["changed_files"],
            commit_message=commit_message,
        )
        emit("branch_pushed", branch=branch_name, head_sha=head_sha)

        base_sha = get_base_sha(repo_root, base_branch)
        pr_number, head_sha = create_pull_request(
            repo_full_name=repo_full_name,
            branch_name=branch_name,
            base_branch=base_branch,
            title=f"[{jira_key}] {dev_result.get('summary', 'Automated change')}",
            body=f"Automated implementation by Dev Agent.\n\nTask:\n{task_text}",
        )
        emit("pull_request_created", number=pr_number, branch=branch_name)
    except Exception as error:
        state = transition(state, WorkflowState.DEV_BLOCKED)
        state = transition(state, WorkflowState.WORKFLOW_FAILED)
        report["final_state"] = state.value
        report["git_error"] = str(error)
        emit("workflow_failed", task_id=task_id, stage="git_or_pull_request", reason=str(error))
        return report

    state = transition(state, WorkflowState.PR_CREATED)
    report["pull_request"] = {
        "number": pr_number,
        "branch": branch_name,
        "head_sha": head_sha,
    }

    # ---- DCBA phase 2: reallocate the review budget from the REAL diff ----
    refined_review_budget = reallocate_review_budget(
        repo_root=repo_root,
        base_sha=base_sha,
        head_sha=head_sha,
        changed_files=dev_result["changed_files"],
        dev_actual_tokens_used=dev_actual_tokens or budgets["dev_agent"],
        total_budget=total_budget,
        review_ema=review_ema,
    )
    budgets["code_review_agent"] = refined_review_budget
    report["dcba_budgets_phase2"] = {"code_review_agent": refined_review_budget}
    emit(
        "dcba_review_budget_refined",
        refined_tokens=refined_review_budget,
        initial_estimate=raw_budgets["code_review_agent"],
        based_on_actual_diff=True,
    )

    state = transition(state, WorkflowState.CODE_REVIEW_IN_PROGRESS)
    emit("code_review_started", pull_request=pr_number, allocated_tokens=budgets["code_review_agent"])

    try:
        from run_review import run_review_for_pr  # noqa: E402

        review_result = run_review_for_pr(
            repo_root=repo_root,
            base_sha=base_sha,
            head_sha=head_sha,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            budget_tokens=budgets["code_review_agent"],
            db_path=db_path,
        )
        review_error = None
    except Exception as error:  # was `except ImportError`, silently faking success
        review_result = {"action": "ERROR"}
        review_error = str(error)

    report["code_review_agent"] = review_result
    decision_action = review_result.get("action")
    emit("code_review_finished", action=decision_action, result=review_result, error=review_error)

    review_actual_tokens = None  # review agent doesn't return usage yet — see note below

    if review_error is not None:
        # Fail closed: review genuinely did not run. Don't pretend it did
        # by routing to HUMAN_APPROVAL_REQUIRED as if a review happened.
        state = transition(state, WorkflowState.CODE_REVIEW_BLOCKED)
        state = transition(state, WorkflowState.WORKFLOW_FAILED)
        emit("workflow_failed", task_id=task_id, stage="code_review", reason=review_error)
    elif decision_action == "REJECT":
        state = transition(state, WorkflowState.CODE_REVIEW_BLOCKED)
        state = transition(state, WorkflowState.WORKFLOW_FAILED)
        emit("workflow_failed", task_id=task_id, stage="code_review", reason="REJECT")
    else:
        target = (
            WorkflowState.CODE_REVIEW_PASSED
            if decision_action == "AUTO_APPROVE"
            else WorkflowState.CODE_REVIEW_BLOCKED
        )
        state = transition(state, target)
        state = transition(state, WorkflowState.HUMAN_APPROVAL_REQUIRED)
        emit("human_approval_required", pull_request=pr_number, review_status=decision_action)

    report["final_state"] = state.value

    try:
        summary_prompt = load_prompt(PROMPTS_DIR, "orchestrator_summary_prompt.md")
        key_pool = GroqKeyPool()
        summary_user_prompt = (
            f"## Task\n{task_text}\n\n"
            f"## Dev Agent result\n{json.dumps(dev_result)}\n\n"
            f"## Code Review Agent result\n{json.dumps(review_result)}\n\n"
            f"## Final workflow state\n{state.value}"
        )
        summary = call_groq_json(
            key_pool=key_pool,
            model=DEFAULT_MODEL,
            system_prompt=summary_prompt,
            user_prompt=summary_user_prompt,
            max_output_tokens=400,
            token_ceiling=2000,
        )
        summary.pop("_usage", None)
        report["human_readable_summary"] = summary
        emit("summary_generated")
    except Exception as error:
        report["human_readable_summary"] = {"summary": f"(summary generation failed: {error})"}
        emit("summary_generation_failed", reason=str(error))

    emit("pipeline_finished", task_id=task_id, final_state=state.value)
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
            task_text=task_text,
            repo_root=args.repo_root,
            db_path=args.db_path,
            repo_full_name=args.repo_full_name,
            base_branch=args.base_branch,
            total_budget=args.total_budget,
        )
    except InvalidTransitionError as error:
        report = {"error": str(error)}

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()