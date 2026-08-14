"""
agents/dev_agent/dev_agent.py  (v2)

Produces code changes only — no git operations here. The orchestrator
(master_agent.py) owns branching/committing/pushing/PR-creation, matching
the separation of concerns you described.

Editing mechanism (see agents/common/patch_apply.py):
    - Existing files: LLM outputs a unified diff -> applied via `git apply`.
    - New files: LLM outputs full content -> written directly.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "knowledge_base"))

from groq_client import GroqKeyPool, call_groq_json, load_prompt  # noqa: E402
from patch_apply import apply_unified_diff, write_full_file  # noqa: E402
from context_selector import select_context  # noqa: E402
from kb_schema import get_connection  # noqa: E402

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def run(
    task_text: str,
    repo_root: str,
    db_path: str,
    allocated_budget_tokens: int,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 1500,
) -> Dict[str, Any]:
    start_time = time.time()

    system_prompt = load_prompt(PROMPTS_DIR, "dev_agent_prompt.md")

    context_budget = int(allocated_budget_tokens * 0.6)
    if os.path.exists(db_path):
        conn = get_connection(db_path)
        kb_context = select_context(conn, task_text, budget_tokens=context_budget)
        conn.close()
    else:
        kb_context = "(knowledge base not yet built — run build_kb.py first)"

    user_prompt = (
        f"## Task\n{task_text}\n\n"
        f"## Relevant existing code (hybrid BM25 + vector + PageRank retrieval)\n{kb_context}"
    )

    key_pool = GroqKeyPool()
    result = call_groq_json(
        key_pool=key_pool, model=model, system_prompt=system_prompt,
        user_prompt=user_prompt, max_output_tokens=max_output_tokens,
        token_ceiling=allocated_budget_tokens,
    )
    usage = result.pop("_usage", {})
    latency_ms = (time.time() - start_time) * 1000

    if result.get("blocked"):
        return {
            "status": "BLOCKED",
            "blocked_reason": result.get("blocked_reason", "unspecified"),
            "usage": usage, "latency_ms": latency_ms,
        }

    diffs = result.get("diffs", {}) or {}
    new_files = result.get("new_files", {}) or {}

    if not diffs and not new_files:
        return {
            "status": "BLOCKED",
            "blocked_reason": "Dev Agent returned no diffs or new files.",
            "usage": usage, "latency_ms": latency_ms,
        }

    changed_files: List[str] = []
    apply_errors: List[str] = []

    for rel_path, content in new_files.items():
        if rel_path.startswith("agents/") or rel_path.startswith(".github/"):
            continue
        write_full_file(repo_root, rel_path, content)
        changed_files.append(rel_path)

    for rel_path, diff_text in diffs.items():
        if rel_path.startswith("agents/") or rel_path.startswith(".github/"):
            continue
        success, message = apply_unified_diff(repo_root, diff_text)
        if success:
            changed_files.append(rel_path)
        else:
            apply_errors.append(f"{rel_path}: {message}")

    if apply_errors and not changed_files:
        return {
            "status": "BLOCKED",
            "blocked_reason": f"All patches failed to apply: {apply_errors}",
            "usage": usage, "latency_ms": latency_ms,
        }

    return {
        "status": "FILES_READY",
        "changed_files": changed_files,
        "apply_errors": apply_errors,
        "summary": result.get("summary", ""),
        "jira_key": result.get("jira_key", ""),
        "usage": usage, "latency_ms": latency_ms,
    }
