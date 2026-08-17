"""
agents/dev_agent/dev_agent.py  (v5)

CHANGES from v4 (both found from a real failure log, not hypothetical):

  1. _guess_target_files() used os.listdir(repo_root) -- NON-RECURSIVE.
     Any target file not sitting directly at the repo root (e.g. inside
     a package subfolder) was invisible to the ground-truth heuristic,
     silently fell back to "(no exact file content available)", and the
     model guessed anyway instead of honoring "set blocked=true" --
     which is exactly what produced a diff that replaced
     `elif choice == "0":` with the identical line, while treating
     print("bye") as already-existing context instead of the actual
     current print("Goodbye."). Fixed: os.walk() over the whole repo
     (excluding platform/junk dirs), matching either the bare filename
     or the full relative path against the task text.

  2. NEW: ground-truth enforcement at apply time, not just prompt time.
     Every existing file the model tries to touch (via new_files OR
     diffs) is now checked against the set of files that were ACTUALLY
     shown to the model as ground truth. If a file exists on disk but
     wasn't in that set, the edit is refused with a clear error instead
     of trusted blindly -- this doesn't rely on the model reliably
     following "don't guess" instructions, which it demonstrably
     doesn't always do.

  3. NEW: _diff_is_meaningful() rejects a diff BEFORE calling git apply
     if its removed and added lines are identical (a no-op edit). This
     specific failure mode is dangerous precisely because a no-op diff
     with a CORRECT hunk header would apply successfully and silently
     do nothing -- the "patch does not apply" error in the log was
     lucky; a slightly different line-number miscount would not have
     been.

Editing mechanism unchanged: diffs -> git apply, new/rewritten files ->
written directly, guarded by _validate_preserves_structure() when the
file already existed.
"""

import difflib
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

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

# Directories the Dev Agent should never scan into: platform code it must
# never touch (agents/, .github/), VCS internals, and junk that could
# make the recursive walk slow or pull in noise.
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
    to the model verbatim rather than left to guesswork.

    Recursive (os.walk), not just the repo root -- a file living in a
    subfolder used to be invisible to this heuristic entirely, which is
    what let a hallucinated rewrite through undetected.

    Deliberately over-inclusive -- a false positive just means one extra
    file gets shown as ground truth (harmless); a false negative is what
    causes hallucinated rewrites."""
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
    The verified set is what apply-time enforcement checks against --
    it's the ground truth for "was this file's real content ever shown
    to the model", independent of what the model claims to have done."""
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

def _files_requiring_full_replacement(target_files: List[str], max_chars: int = GROUND_TRUTH_MAX_CHARS_PER_FILE) -> Set[str]:
    """Any file we supplied as ground truth is small enough that a full
    replacement is cheap and dramatically more reliable than a unified
    diff from this model. Diffs on these files are now rejected outright
    rather than sent to git apply -- three separate real failures (wrong
    line targeted, synthetic no-op hunk, and context/indentation mismatch)
    show unified-diff generation is not reliable enough to trust here."""
    return set(target_files)

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


def _diff_is_meaningful(diff_text: str) -> bool:
    """Reject a diff whose hunks contain no actual content change -- every
    removed line has an identical added-line counterpart (as a multiset,
    so reordering doesn't fool it). This is exactly the failure seen in
    production: a hunk that replaces `elif choice == "0":` with the
    identical line. Rejecting this BEFORE calling git apply matters
    because a no-op diff with a numerically-correct hunk header would
    apply cleanly and silently do nothing -- catching it only via a
    'patch does not apply' error is luck, not a guarantee."""
    removed = [line[1:] for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---")]
    added = [line[1:] for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++")]
    if not removed and not added:
        return False
    return Counter(removed) != Counter(added)


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
    ground_truth_block, verified_ground_truth_files = _build_ground_truth_block(repo_root, target_files)


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
    ground_truth_files = set(target_files)

    for rel_path, diff_text in diffs.items():
        if rel_path.startswith("agents/") or rel_path.startswith(".github/"):
            continue

        if rel_path in ground_truth_files:
            apply_errors.append(
                f"{rel_path}: diffs are not permitted for files supplied as "
                f"ground truth -- resubmit using new_files with the complete "
                f"file content instead. This file's diff was rejected before "
                f"being sent to git apply."
            )
            continue

        success, message = apply_unified_diff(repo_root, diff_text)
        if success:
            changed_files.append(rel_path)
        else:
            apply_errors.append(f"{rel_path}: {message}")
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

        if existing_content is not None and rel_path not in verified_ground_truth_files:
            apply_errors.append(
                f"{rel_path}: refusing to overwrite -- this file exists but its exact "
                f"content was never shown to the model as ground truth (missed by the "
                f"task-text file heuristic), so a full-file replacement here would be "
                f"an unverified guess."
            )
            continue

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

        existing_content = _read_existing_file(repo_root, rel_path, max_chars=1_000_000)
        if existing_content is not None and rel_path not in verified_ground_truth_files:
            apply_errors.append(
                f"{rel_path}: refusing to apply diff -- this file's exact content was "
                f"never shown to the model as ground truth, so the diff's context lines "
                f"can't be trusted to match the real file."
            )
            continue

        if not _diff_is_meaningful(diff_text):
            apply_errors.append(
                f"{rel_path}: diff rejected -- added and removed lines are identical "
                f"(no-op edit). This usually means the model didn't actually have the "
                f"real file content and hallucinated a plausible-looking but empty change."
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