"""
agents/orchestrator/state_machine.py

Deterministic workflow state machine. This — not any LLM's "reasoning" —
is what decides whether the pipeline advances, retries, or stops. The
orchestrator's LLM call (orchestrator_summary_prompt.md) only explains
results after the fact; it never controls transitions.
"""

from enum import Enum
from typing import Dict, Set


class WorkflowState(str, Enum):
    TASK_RECEIVED = "TASK_RECEIVED"
    DEV_IN_PROGRESS = "DEV_IN_PROGRESS"
    DEV_BLOCKED = "DEV_BLOCKED"
    PR_CREATED = "PR_CREATED"
    CODE_REVIEW_IN_PROGRESS = "CODE_REVIEW_IN_PROGRESS"
    CODE_REVIEW_PASSED = "CODE_REVIEW_PASSED"
    CODE_REVIEW_BLOCKED = "CODE_REVIEW_BLOCKED"
    HUMAN_APPROVAL_REQUIRED = "HUMAN_APPROVAL_REQUIRED"
    WORKFLOW_FAILED = "WORKFLOW_FAILED"


ALLOWED_TRANSITIONS: Dict[WorkflowState, Set[WorkflowState]] = {
    WorkflowState.TASK_RECEIVED: {WorkflowState.DEV_IN_PROGRESS},
    WorkflowState.DEV_IN_PROGRESS: {WorkflowState.PR_CREATED, WorkflowState.DEV_BLOCKED},
    WorkflowState.DEV_BLOCKED: {WorkflowState.WORKFLOW_FAILED},
    WorkflowState.PR_CREATED: {WorkflowState.CODE_REVIEW_IN_PROGRESS},
    WorkflowState.CODE_REVIEW_IN_PROGRESS: {
        WorkflowState.CODE_REVIEW_PASSED,
        WorkflowState.CODE_REVIEW_BLOCKED,
    },
    WorkflowState.CODE_REVIEW_PASSED: {WorkflowState.HUMAN_APPROVAL_REQUIRED},
    WorkflowState.CODE_REVIEW_BLOCKED: {
        WorkflowState.HUMAN_APPROVAL_REQUIRED,  # HUMAN_REVIEW findings still go to a human
        WorkflowState.WORKFLOW_FAILED,           # REJECT-level findings stop the pipeline
    },
    WorkflowState.HUMAN_APPROVAL_REQUIRED: set(),  # terminal — a human decides next, not code
    WorkflowState.WORKFLOW_FAILED: set(),          # terminal
}


class InvalidTransitionError(Exception):
    pass


def transition(current: WorkflowState, target: WorkflowState) -> WorkflowState:
    allowed = ALLOWED_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise InvalidTransitionError(
            f"Cannot transition from {current} to {target}. Allowed: {allowed}"
        )
    return target


def is_terminal(state: WorkflowState) -> bool:
    return len(ALLOWED_TRANSITIONS.get(state, set())) == 0
