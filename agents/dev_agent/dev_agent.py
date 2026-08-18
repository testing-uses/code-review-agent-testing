"""
agents/dev_agent/dev_agent.py (v7)

Produces code changes only -- no git operations here. The orchestrator
(master_agent.py) owns branching/committing/pushing/PR-creation.

CHANGE from v6 (root-caused from a real preflight failure log):

The v6 defaults could never fit inside a realistic DCBA-allocated ceiling:
system prompt (~2,325 tokens) + a full ground-truth file (~1,800 tokens)
+ max_output_tokens (2,500) already exceeds a ~5,000 token ceiling before
KB context or the task text are even added. Two changes fix this without
touching output reliability:

1. KB context is now SKIPPED ENTIRELY whenever ground truth already found
   the target file(s). Every real run logged so far shows
   matched_context_entries=0 -- the KB signature search has never once
   contributed anything useful for these tasks, so spending part of the
   budget on it when we already have the exact file is pure waste. KB
   context is only requested when NO ground-truth file was found, as a
   fallback for the model to at least have something.

2. If preflight_check still rejects the prompt as too large, retry ONCE
   with a compact prompt (KB context dropped, ground truth kept -- ground
   truth is what actually lets the model do a safe edit) BEFORE giving up.
   This retry no longer shrinks max_output_tokens (a prior version did,
   which caused a *different* real failure: the model's output got
   truncated mid-JSON because the line-array format needs more output
   tokens per line than a single string, not fewer).

max_output_tokens default lowered from 2500 to 1800 -- calibrated against
the real truncation and success logs seen so far, this is enough for a
~50-line file in line-array format with escaping overhead, without
needlessly inflating the ceiling DCBA has to clear.

Everything else carried over from v6, unchanged:
- diffs are rejected for any file supplied as ground truth; must use
  new_files with full content instead.
- call_groq_json() BadRequestError (json_validate_failed) is caught and
  converted to a clean BLOCKED result instead of crashing the pipeline.
- new_files values may be a JSON array of lines (preferred) or a plain
  string (backward compatible), normalized by _normalize_file_content().
- _guess_target_files() recursively walks the whole repo.
- Ground-truth enforcement at APPLY time, not just prompt time.
- _diff_is_meaningful() rejects no-op diffs before calling git apply.
- _validate_preserves_structure() rejects hallucinated full rewrites.

Editing mechanism:
- Files supplied as ground truth: MUST use new_files (full content),
  never diffs. Guarded by _validate_preserves_structure().
- Existing files NOT supplied as ground truth: diffs only, applied via
  `git apply`, guarded by _diff_is_meaningful().
- Brand-new files: new_files, written directly.
"""

import difflib
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

_AGENTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_AGENTS_ROOT, "common"))
from path_bootstrap import bootstrap  # noqa: E402
bootstrap()

from groq import BadRequestError  # noqa: E402
from groq_client import GroqKeyPool, call_groq_json, load_prompt  # noqa: E402
from patch_apply import apply_unified_diff, write_full_file  # noqa: E402
from context_selector import select_context  # noqa: E402
from kb_schema import get_connection  # noqa: E402

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
DEFAULT_MODEL = "openai/gpt-oss-120b"

GROUND_TRUTH_MAX_CHARS_PER_FILE = 6000
GROUND_TRUTH_MAX_FILES = 4
REWRITE_SIMILARITY_FLOOR = 0.5

_EXCLUDED_DIR_NAMES = {
    ".git", ".github", "agents", "__pycache__", "node_modules",
    ".venv", "venv", ".review_agent_cache", ".mypy_cache", ".pytest_cache",
}


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
    """Heuristic: any .py file anywhere in the repo whose bare filename OR
    full relative path is literally named in the task text is almost
    certainly a file the Dev Agent is about to touch, and MUST be shown
    to the model verbatim rather than left to guesswork. Recursive
    (os.walk), deliberately over-inclusive."""
    candidates: List[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIR_NAMES and not d.startswith(".")]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            rel_path = os.path.relpath(os.path.join(dirpath, filename), repo_root).replace(os.sep, "/")
            if filename in task_text or rel_path in task_text:
                candidates.append(rel_path)
            if len(candidates) >= GROUND_TRUTH_MAX_FILES:
                return candidates
    return candidates


def _build_ground_truth_block(repo_root: str, target_files: List[str]) -> Tuple[str, Set[str]]:
    """Returns (rendered prompt block, set of rel_paths actually verified).
    The verified set is what apply-time enforcement checks against."""
    blocks = []
    verified: Set[str] = set()
    for rel_path in target_files:
        content = _read_existing_file(repo_root, rel_path)
        if content is None:
            continue
        verified.add(rel_path)
        blocks.append(
            f"### EXACT CURRENT CONTENT of {rel_path} (ground truth -- do not deviate)\n"
            f"```python\n{content}\n```"
        )
    if not blocks:
        return "(no exact file content available -- do not guess at file structure; set blocked=true instead)", verified
    return "\n\n".join(blocks), verified


def _validate_preserves_structure(original: str, rewritten: str, min_similarity: float = REWRITE_SIMILARITY_FLOOR) -> bool:
    """A legitimate small edit should be near-identical to the original.
    A hallucinated from-scratch rewrite will score low."""
    if not original:
        return True
    ratio = difflib.SequenceMatcher(None, original, rewritten).ratio()
    return ratio >= min_similarity


def _diff_is_meaningful(diff_text: str) -> bool:
    """Reject a diff whose hunks contain no actual content change."""
    removed = [line[1:] for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---")]
    added = [line[1:] for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++")]
    if not removed and not added:
        return False
    return Counter(removed) != Counter(added)


def _normalize_file_content(raw_content: Any) -> str:
    """new_files values should be a JSON array of lines -- easier for the
    model to escape correctly than one giant multi-line string. Falls
    back to treating the value as a plain string for compatibility."""
    if isinstance(raw_content, list):
        return "\n".join(str(line) for line in raw_content)
    return str(raw_content)


def _build_user_prompt(task_text: str, kb_context: str, ground_truth_block: str, include_kb_context: bool) -> str:
    kb_section = (
        f"## Relevant existing code (hybrid BM25 + vector + PageRank retrieval -- signatures only, NOT ground truth)\n"
        f"{kb_context}\n\n"
        if include_kb_context else ""
    )
    return (
        f"## Task\n{task_text}\n\n"
        f"{kb_section}"
        f"## Exact current file content -- this IS ground truth\n"
        f"You MUST preserve every line of the files below exactly, except for the minimal\n"
        f"change the task explicitly requires. Do not invent a different program structure,\n"
        f"remove existing menu options/imports/business logic, or 'clean up' unrelated code.\n"
        f"Files shown here MUST be returned via new_files (full content), never diffs.\n"
        f"If a file you need to modify is not shown here, set blocked=true and explain why,\n"
        f"instead of guessing at its contents.\n\n"
        f"{ground_truth_block}"
    )


def run(
    task_text: str,
    repo_root: str,
    db_path: str,
    allocated_budget_tokens: int,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 1800,
) -> Dict[str, Any]:
    start_time = time.time()

    system_prompt = load_prompt(PROMPTS_DIR, "dev_agent_prompt.md")

    target_files = _guess_target_files(task_text, repo_root)
    ground_truth_block, verified_ground_truth_files = _build_ground_truth_block(repo_root, target_files)
    ground_truth_found = bool(verified_ground_truth_files)

    # Skip the KB lookup entirely when ground truth already covers the
    # target file(s) -- it has never once contributed anything useful in
    # practice (matched_context_entries has been 0 on every real run so
    # far) and it only costs budget we need for the exact file content.
    if ground_truth_found:
        kb_context = "(skipped -- ground truth already found for the target file(s))"
    elif os.path.exists(db_path):
        context_budget = int(allocated_budget_tokens * 0.6)
        conn = get_connection(db_path)
        kb_context = select_context(conn, task_text, budget_tokens=context_budget)
        conn.close()
    else:
        kb_context = "(knowledge base not yet built -- run build_kb.py first)"

    user_prompt = _build_user_prompt(
        task_text, kb_context, ground_truth_block,
        include_kb_context=not ground_truth_found,
    )

    key_pool = GroqKeyPool()

    try:
        result = call_groq_json(
            key_pool=key_pool, model=model, system_prompt=system_prompt,
            user_prompt=user_prompt, max_output_tokens=max_output_tokens,
            token_ceiling=allocated_budget_tokens,
        )
    except ValueError as error:
        if "Prompt too large" not in str(error):
            raise

        # Last resort: drop KB context entirely (even if it was already
        # excluded) and keep only the task + ground truth. Do NOT shrink
        # max_output_tokens here -- that caused a separate real failure
        # (truncated mid-JSON output) in an earlier version.
        compact_user_prompt = _build_user_prompt(
            task_text, kb_context="", ground_truth_block=ground_truth_block,
            include_kb_context=False,
        )

        try:
            result = call_groq_json(
                key_pool=key_pool, model=model, system_prompt=system_prompt,
                user_prompt=compact_user_prompt, max_output_tokens=max_output_tokens,
                token_ceiling=allocated_budget_tokens,
            )
        except ValueError as retry_error:
            latency_ms = (time.time() - start_time) * 1000
            return {
                "status": "BLOCKED",
                "blocked_reason": (
                    f"Prompt still too large after dropping KB context: {retry_error}. "
                    f"Raise DCBA's DEFAULT_TOTAL_BUDGET/MIN_BUDGET_PER_AGENT or reduce "
                    f"GROUND_TRUTH_MAX_CHARS_PER_FILE."
                ),
                "usage": {}, "latency_ms": latency_ms,
            }
    except BadRequestError as error:
        latency_ms = (time.time() - start_time) * 1000
        return {
            "status": "BLOCKED",
            "blocked_reason": (
                "Groq rejected the generation as invalid JSON (the model likely "
                f"broke JSON escaping while embedding a multi-line file with "
                f"quotes/docstrings). Raw error: {error}"
            ),
            "usage": {}, "latency_ms": latency_ms,
        }

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
    ground_truth_files = set(target_files)

    for rel_path, raw_content in new_files.items():
        if rel_path.startswith("agents/") or rel_path.startswith(".github/"):
            continue

        content = _normalize_file_content(raw_content)
        existing_content = _read_existing_file(repo_root, rel_path, max_chars=1_000_000)

        if existing_content is not None and rel_path not in verified_ground_truth_files:
            apply_errors.append(
                f"{rel_path}: refusing to overwrite -- this file exists but its exact "
                f"content was never shown to the model as ground truth."
            )
            continue

        if existing_content is not None and not _validate_preserves_structure(existing_content, content):
            apply_errors.append(
                f"{rel_path}: rewrite rejected -- new content is too dissimilar "
                f"to the existing file (looks like a hallucinated full rewrite)."
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

        if rel_path in ground_truth_files:
            apply_errors.append(
                f"{rel_path}: diffs are not permitted for files supplied as "
                f"ground truth -- resubmit using new_files with the complete "
                f"file content instead."
            )
            continue

        existing_content = _read_existing_file(repo_root, rel_path, max_chars=1_000_000)
        if existing_content is not None and rel_path not in verified_ground_truth_files:
            apply_errors.append(
                f"{rel_path}: refusing to apply diff -- this file's exact content was "
                f"never shown to the model as ground truth."
            )
            continue

        if not _diff_is_meaningful(diff_text):
            apply_errors.append(
                f"{rel_path}: diff rejected -- added and removed lines are identical "
                f"(no-op edit)."
            )
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