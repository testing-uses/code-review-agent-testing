"""
agents/dev_agent/dev_agent.py  (v4)

Produces code changes only -- no git operations here. The orchestrator
(master_agent.py) owns branching/committing/pushing/PR-creation.

CHANGE from v3: fixes the "Dev Agent wipes the whole file and writes a
half-baked generic replacement" bug.

Root cause: the KB's context_selector only returns symbol SIGNATURES and
short docstrings, never actual file content. When a task asked for a
full-file rewrite (new_files), the model had nothing real to reproduce,
so it hallucinated a plausible-looking but completely different file
(e.g. replacing a LibraryStore-backed CLI with a generic "Option 1 /
Option 2 / Quit" menu).

Two independent fixes, both required:
    1. GROUND TRUTH INJECTION -- before calling the model, read the exact
       current content of any file plausibly targeted by the task (any
       .py filename literally mentioned in the task text) directly off
       disk, and inject it into the user prompt as non-negotiable ground
       truth, separate from the KB's symbol-level context.
    2. REWRITE GUARD -- after the model responds, if it returned a
       new_files entry for a file that already exists on disk, compare
       the new content against the original with difflib. If similarity
       falls below a threshold, treat it as a hallucinated rewrite and
       BLOCK instead of silently committing it. A legitimate small
       change to an existing file should score very high; a full
       reinvention scores low.

Editing mechanism (see agents/common/patch_apply.py):
    - Existing files: LLM outputs a unified diff -> applied via `git apply`.
    - New/rewritten files: LLM outputs full content -> written directly,
      but guarded by _validate_preserves_structure() when the file
      already existed.
"""

import difflib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

_AGENTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_AGENTS_ROOT, "common"))
from path_bootstrap import bootstrap  # noqa: E402
bootstrap()

from groq_client import GroqKeyPool, call_groq_json, load_prompt  # noqa: E402
from patch_apply import apply_unified_diff, write_full_file  # noqa: E402
from context_selector import select_context  # noqa: E402
from kb_schema import get_connection  # noqa: E402

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
DEFAULT_MODEL = "llama-3.3-70b-versatile"

GROUND_TRUTH_MAX_CHARS_PER_FILE = 6000
GROUND_TRUTH_MAX_FILES = 4
REWRITE_SIMILARITY_FLOOR = 0.5


def _read_existing_file(repo_root: str, rel_path: str, max_chars: int = GROUND_TRUTH_MAX_CHARS_PER_FILE) -> Optional[str]:
    full_path = os.path.join(repo_root, rel_path)
    if not os.path.isfile(full_path):
        return None
    try:
        with open(full_path, "r", encoding="utf-8") as fh:
            content = fh.read()
    except (OSError, UnicodeDecodeError):
        return None
    if len(content) > max_chars:
        content = content[:max_chars] + "\n...[truncated -- file longer than ground-truth limit]..."
    return content


def _guess_target_files(task_text: str, repo_root: str) -> List[str]:
    """Heuristic: any top-level .py filename literally named in the task
    text is almost certainly a file the Dev Agent is about to touch, and
    MUST be shown to the model verbatim rather than left to guesswork.
    Deliberately simple and over-inclusive -- a false positive just means
    one extra file gets included in the prompt as ground truth, which is
    harmless; a false negative is what causes hallucinated rewrites."""
    candidates = []
    for entry in sorted(os.listdir(repo_root)):
        full_path = os.path.join(repo_root, entry)
        if not os.path.isfile(full_path):
            continue
        if not entry.endswith(".py"):
            continue
        if entry in task_text:
            candidates.append(entry)
    return candidates[:GROUND_TRUTH_MAX_FILES]


def _build_ground_truth_block(repo_root: str, target_files: List[str]) -> str:
    blocks = []
    for rel_path in target_files:
        content = _read_existing_file(repo_root, rel_path)
        if content is None:
            continue
        blocks.append(
            f"### EXACT CURRENT CONTENT of {rel_path} (ground truth -- do not deviate)\n"
            f"```python\n{content}\n```"
        )
    if not blocks:
        return "(no exact file content available -- do not guess at file structure; set blocked=true instead)"
    return "\n\n".join(blocks)


def _validate_preserves_structure(original: str, rewritten: str, min_similarity: float = REWRITE_SIMILARITY_FLOOR) -> bool:
    """A legitimate small edit to an existing file should be near-identical
    to the original. A hallucinated from-scratch rewrite will score low.
    This is a coarse, deliberately conservative safety net -- it can't
    tell you the edit is CORRECT, only that it isn't a wholesale
    replacement of the file's actual structure."""
    if not original:
        return True
    ratio = difflib.SequenceMatcher(None, original, rewritten).ratio()
    return ratio >= min_similarity


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
        kb_context = "(knowledge base not yet built -- run build_kb.py first)"

    target_files = _guess_target_files(task_text, repo_root)
    ground_truth_block = _build_ground_truth_block(repo_root, target_files)

    user_prompt = (
        f"## Task\n{task_text}\n\n"
        f"## Relevant existing code (hybrid BM25 + vector + PageRank retrieval -- signatures only, NOT ground truth)\n"
        f"{kb_context}\n\n"
        f"## Exact current file content -- this IS ground truth\n"
        f"You MUST preserve every line of the files below exactly, except for the minimal\n"
        f"change the task explicitly requires. Do not invent a different program structure,\n"
        f"remove existing menu options/imports/business logic, or 'clean up' unrelated code.\n"
        f"If a file you need to modify is not shown here, set blocked=true and explain why,\n"
        f"instead of guessing at its contents.\n\n"
        f"{ground_truth_block}"
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

        existing_content = _read_existing_file(repo_root, rel_path, max_chars=1_000_000)

        if existing_content is not None:
            if not _validate_preserves_structure(existing_content, content):
                apply_errors.append(
                    f"{rel_path}: rewrite rejected -- new content is too dissimilar "
                    f"to the existing file (looks like a hallucinated full rewrite "
                    f"rather than a targeted edit). Re-run with a more explicit task "
                    f"or ensure ground-truth content was supplied."
                )
                continue

        try:
            write_full_file(repo_root, rel_path, content)
            changed_files.append(rel_path)
        except ValueError as error:
            apply_errors.append(f"{rel_path}: {error}")

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