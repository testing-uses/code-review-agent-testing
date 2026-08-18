"""
agents/dev_agent/dev_agent.py (v6)

Produces code changes only -- no git operations here. The orchestrator
(master_agent.py) owns branching/committing/pushing/PR-creation.

CHANGES from v5 (both found from real failure logs, not hypothetical):

1. NEW: diffs are now rejected outright for any file supplied as ground
   truth, and must instead be returned via new_files with the complete
   content. Three separate real failures (wrong line targeted, synthetic
   no-op hunk, context/indentation mismatch) showed unified-diff
   generation from this model is not reliable enough to trust for small
   files we already have exact content for. Full-file replacement is
   protected by _validate_preserves_structure() instead of git apply's
   strict line-count/context matching.

2. NEW: call_groq_json() is now wrapped in a try/except for Groq's own
   BadRequestError (json_validate_failed). A real run crashed the entire
   GitHub Actions job with an unhandled traceback because the model tried
   to embed a full Python file -- including its own triple-quoted
   docstring -- as a single JSON string and got the escaping wrong. This
   is now caught and converted into a clean BLOCKED result instead of a
   pipeline crash.

3. NEW: new_files content may now be a JSON array of lines instead of one
   big string with embedded newlines/quotes. This is what actually
   prevents the JSON-escaping crash in (2) from happening in the first
   place -- per-line strings are trivial for the model to escape
   correctly; one giant multi-line string containing its own quotes and
   triple-quotes is not. _normalize_file_content() joins the array with
   "\n"; a plain string is still accepted for backward compatibility.

Carried over from v5 (still required, still correct):
    - _guess_target_files() recursively walks the whole repo (os.walk),
      not just the repo root, so files in subfolders aren't invisible to
      the ground-truth heuristic.
    - Ground-truth enforcement at APPLY time, not just prompt time: any
      existing file the model touches must have been actually shown to it
      as ground truth, regardless of what the model claims.
    - _diff_is_meaningful() rejects a no-op diff (identical removed/added
      lines) before calling git apply.
    - _validate_preserves_structure() rejects a new_files rewrite of an
      existing file if it's too dissimilar to the original (hallucinated
      rewrite guard).

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

# Directories the Dev Agent should never scan into: platform code it must
# never touch (agents/, .github/), VCS internals, and junk that could make
# the recursive walk slow or pull in noise.
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
    subfolder used to be invisible to this heuristic entirely.

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
    The verified set is what apply-time enforcement checks against -- it's
    the ground truth for "was this file's real content ever shown to the
    model", independent of what the model claims to have done."""
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
    so reordering doesn't fool it). A no-op diff with a numerically
    correct hunk header would apply cleanly and silently do nothing --
    catching it only via a 'patch does not apply' error is luck, not a
    guarantee."""
    removed = [line[1:] for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---")]
    added = [line[1:] for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++")]
    if not removed and not added:
        return False
    return Counter(removed) != Counter(added)


def _normalize_file_content(raw_content: Any) -> str:
    """new_files values should be a JSON array of lines -- this avoids the
    model having to hand-escape a multi-line string containing its own
    quotes/triple-quotes, which is exactly what caused a real Groq-side
    400 json_validate_failed error (the model tried to embed a Python
    docstring's literal \"\"\" inside one giant JSON string and broke
    escaping). Falls back to treating the value as a plain string for
    backward compatibility with responses that don't use the array form."""
    if isinstance(raw_content, list):
        return "\n".join(str(line) for line in raw_content)
    return str(raw_content)


def run(
    task_text: str,
    repo_root: str,
    db_path: str,
    allocated_budget_tokens: int,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 2500,
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
        f"Files shown here MUST be returned via new_files (full content), never diffs.\n"
        f"If a file you need to modify is not shown here, set blocked=true and explain why,\n"
        f"instead of guessing at its contents.\n\n"
        f"{ground_truth_block}"
    )

    key_pool = GroqKeyPool()

    try:
        result = call_groq_json(
            key_pool=key_pool, model=model, system_prompt=system_prompt,
            user_prompt=user_prompt, max_output_tokens=max_output_tokens,
            token_ceiling=allocated_budget_tokens,
        )
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

    # ---- new_files: brand-new files, or full-content edits of files that
    # were shown to the model as ground truth ----
    for rel_path, raw_content in new_files.items():
        if rel_path.startswith("agents/") or rel_path.startswith(".github/"):
            continue

        content = _normalize_file_content(raw_content)
        existing_content = _read_existing_file(repo_root, rel_path, max_chars=1_000_000)

        if existing_content is not None and rel_path not in verified_ground_truth_files:
            apply_errors.append(
                f"{rel_path}: refusing to overwrite -- this file exists but its exact "
                f"content was never shown to the model as ground truth (missed by the "
                f"task-text file heuristic), so a full-file replacement here would be "
                f"an unverified guess."
            )
            continue

        if existing_content is not None and not _validate_preserves_structure(existing_content, content):
            apply_errors.append(
                f"{rel_path}: rewrite rejected -- new content is too dissimilar "
                f"to the existing file (looks like a hallucinated full rewrite "
                f"rather than a targeted edit)."
            )
            continue

        try:
            write_full_file(repo_root, rel_path, content)
            changed_files.append(rel_path)
        except ValueError as error:
            apply_errors.append(f"{rel_path}: {error}")

    # ---- diffs: only for existing files NOT supplied as ground truth ----
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