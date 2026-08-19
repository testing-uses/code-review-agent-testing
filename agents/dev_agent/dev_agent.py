"""agents/dev_agent/dev_agent.py

The Dev Agent asks the model for a minimal repository edit, validates the
response, and applies only safe changes. Existing files that are identified
from the task are supplied as exact ground truth to the model. Full-file
replacements are used for those files because they are more reliable than
model-generated unified diffs for small targeted edits.
"""

import difflib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

_AGENTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_AGENTS_ROOT, "common"))

from path_bootstrap import bootstrap  # noqa: E402

bootstrap()

from context_selector import select_context  # noqa: E402
from groq_client import GroqKeyPool, call_groq_json, load_prompt  # noqa: E402
from kb_schema import get_connection  # noqa: E402
from patch_apply import apply_unified_diff, write_full_file  # noqa: E402

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
DEFAULT_MODEL = "llama-3.3-70b-versatile"

GROUND_TRUTH_MAX_CHARS_PER_FILE = 6000
GROUND_TRUTH_MAX_FILES = 4
REWRITE_SIMILARITY_FLOOR = 0.5

_EXCLUDED_DIR_NAMES = {
    ".git",
    ".github",
    "agents",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".review_agent_cache",
    ".mypy_cache",
    ".pytest_cache",
}


def _read_existing_file(
    repo_root: str,
    rel_path: str,
    max_chars: int = GROUND_TRUTH_MAX_CHARS_PER_FILE,
) -> Optional[str]:
    """Read a repository file, optionally truncating it for model context."""
    full_path = os.path.join(repo_root, rel_path)

    if not os.path.isfile(full_path):
        return None

    try:
        with open(full_path, "r", encoding="utf-8") as file_handle:
            content = file_handle.read()
    except (OSError, UnicodeDecodeError):
        return None

    if len(content) > max_chars:
        content = (
            content[:max_chars]
            + "\n...[truncated -- file longer than ground-truth limit]..."
        )

    return content


def _guess_target_files(task_text: str, repo_root: str) -> List[str]:
    """Find Python files explicitly mentioned by filename or relative path.

    The search is recursive but skips the agent platform itself and common
    generated/dependency directories. A false positive only adds context;
    a false negative permits the model to guess, so this is intentionally
    conservative.
    """
    candidates: List[str] = []
    normalized_task = task_text.replace("\\", "/")

    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [
            directory
            for directory in dirnames
            if directory not in _EXCLUDED_DIR_NAMES
            and not directory.startswith(".")
        ]

        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue

            rel_path = os.path.relpath(
                os.path.join(dirpath, filename),
                repo_root,
            ).replace(os.sep, "/")

            if filename in normalized_task or rel_path in normalized_task:
                candidates.append(rel_path)

                if len(candidates) >= GROUND_TRUTH_MAX_FILES:
                    return candidates

    return candidates


def _build_ground_truth_block(
    repo_root: str,
    target_files: List[str],
) -> Tuple[str, Set[str]]:
    """Build exact-content prompt context and return verified file paths."""
    blocks: List[str] = []
    verified: Set[str] = set()

    for rel_path in target_files:
        content = _read_existing_file(repo_root, rel_path)

        if content is None:
            continue

        verified.add(rel_path)
        blocks.append(
            f"### EXACT CURRENT CONTENT of {rel_path} "
            "(ground truth -- do not deviate)\n"
            f"```python\n{content}\n```"
        )

    if not blocks:
        return (
            "(no exact file content available -- do not guess at file "
            "structure; set blocked=true instead)",
            verified,
        )

    return "\n\n".join(blocks), verified


def _validate_preserves_structure(
    original: str,
    rewritten: str,
    min_similarity: float = REWRITE_SIMILARITY_FLOOR,
) -> bool:
    """Reject likely wholesale hallucinated rewrites of existing files."""
    if not original:
        return True

    ratio = difflib.SequenceMatcher(None, original, rewritten).ratio()
    return ratio >= min_similarity


def _diff_is_meaningful(diff_text: str) -> bool:
    """Reject an empty or no-op unified diff before invoking git apply."""
    removed = [
        line[1:]
        for line in diff_text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    added = [
        line[1:]
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]

    if not removed and not added:
        return False

    return removed != added


def _normalize_model_result(result: Any) -> Dict[str, Any]:
    """Ensure model output is a dictionary with predictable edit fields."""
    if not isinstance(result, dict):
        return {
            "blocked": True,
            "blocked_reason": "Model returned a non-object JSON response.",
            "raw_result": result,
            "diffs": {},
            "new_files": {},
        }

    normalized = dict(result)
    diffs = normalized.get("diffs", {}) or {}
    new_files = normalized.get("new_files", {}) or {}

    normalized["diffs"] = diffs if isinstance(diffs, dict) else {}
    normalized["new_files"] = new_files if isinstance(new_files, dict) else {}

    return normalized


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
        connection = get_connection(db_path)
        try:
            kb_context = select_context(
                connection,
                task_text,
                budget_tokens=context_budget,
            )
        finally:
            connection.close()
    else:
        kb_context = "(knowledge base not yet built -- run build_kb.py first)"

    target_files = _guess_target_files(task_text, repo_root)
    ground_truth_block, verified_ground_truth_files = _build_ground_truth_block(
        repo_root,
        target_files,
    )

    user_prompt = (
        f"## Task\n{task_text}\n\n"
        "## Relevant existing code\n"
        "The following retrieval context is for orientation only. It is not "
        "ground truth.\n"
        f"{kb_context}\n\n"
        "## Exact current file content -- this IS ground truth\n"
        "Make only the smallest change explicitly requested by the task. "
        "Preserve every other line, import, function, option, and behavior. "
        "Do not clean up unrelated code. If the required file is not shown, "
        "set blocked=true and explain why instead of guessing.\n\n"
        f"{ground_truth_block}\n\n"
        "## Required response format\n"
        "Return one JSON object only. Use this shape:\n"
        '{"blocked": false, "blocked_reason": "", '
        '"summary": "...", "jira_key": "DEV", '
        '"new_files": {"relative/path.py": "complete content"}, '
        '"diffs": {}}\n'
        "For an existing file supplied as ground truth, use new_files with "
        "the complete updated file content. Do not return a no-op edit. "
        "If you cannot make a safe, targeted edit, return blocked=true and "
        "include a precise blocked_reason."
    )

    key_pool = GroqKeyPool()
    raw_result = call_groq_json(
        key_pool=key_pool,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_output_tokens=max_output_tokens,
        token_ceiling=allocated_budget_tokens,
    )

    usage = raw_result.pop("_usage", {}) if isinstance(raw_result, dict) else {}
    result = _normalize_model_result(raw_result)
    latency_ms = (time.time() - start_time) * 1000

    if result.get("blocked"):
        return {
            "status": "BLOCKED",
            "blocked_reason": result.get(
                "blocked_reason",
                "Dev Agent blocked the task without a reason.",
            ),
            "model_result": result,
            "usage": usage,
            "latency_ms": latency_ms,
        }

    # These assignments deliberately happen before every .items() call.
    diffs = result["diffs"]
    new_files = result["new_files"]

    if not diffs and not new_files:
        return {
            "status": "BLOCKED",
            "blocked_reason": (
                "Dev Agent returned no diffs or new files. "
                f"Model response keys: {sorted(result.keys())}. "
                f"blocked={result.get('blocked')!r}. "
                f"blocked_reason={result.get('blocked_reason')!r}. "
                f"summary={result.get('summary')!r}."
            ),
            "model_result": result,
            "usage": usage,
            "latency_ms": latency_ms,
        }

    changed_files: List[str] = []
    apply_errors: List[str] = []

    # Apply complete-file responses first.
    for rel_path, content in new_files.items():
        if not isinstance(rel_path, str) or not isinstance(content, str):
            apply_errors.append(
                "new_files contains a non-string path or file content."
            )
            continue

        rel_path = rel_path.replace("\\", "/")

        if rel_path.startswith("agents/") or rel_path.startswith(".github/"):
            apply_errors.append(
                f"{rel_path}: edits to protected agent/workflow files are "
                "not permitted by the Dev Agent."
            )
            continue

        existing_content = _read_existing_file(
            repo_root,
            rel_path,
            max_chars=1_000_000,
        )

        if existing_content is not None and rel_path not in verified_ground_truth_files:
            apply_errors.append(
                f"{rel_path}: refusing to overwrite -- this file exists but "
                "its exact content was not shown to the model as ground truth."
            )
            continue

        if existing_content is not None and not _validate_preserves_structure(
            existing_content,
            content,
        ):
            apply_errors.append(
                f"{rel_path}: rewrite rejected -- new content is too dissimilar "
                "to the existing file and looks like a wholesale hallucinated "
                "rewrite."
            )
            continue

        try:
            write_full_file(repo_root, rel_path, content)
            changed_files.append(rel_path)
        except (OSError, ValueError) as error:
            apply_errors.append(f"{rel_path}: {error}")

    # Apply diffs only for files that were not supplied as exact ground truth.
    for rel_path, diff_text in diffs.items():
        if not isinstance(rel_path, str) or not isinstance(diff_text, str):
            apply_errors.append(
                "diffs contains a non-string path or diff content."
            )
            continue

        rel_path = rel_path.replace("\\", "/")

        if rel_path.startswith("agents/") or rel_path.startswith(".github/"):
            apply_errors.append(
                f"{rel_path}: edits to protected agent/workflow files are "
                "not permitted by the Dev Agent."
            )
            continue

        existing_content = _read_existing_file(
            repo_root,
            rel_path,
            max_chars=1_000_000,
        )

        if existing_content is not None and rel_path not in verified_ground_truth_files:
            apply_errors.append(
                f"{rel_path}: refusing to apply diff -- this file's exact "
                "content was not shown to the model as ground truth."
            )
            continue

        if not _diff_is_meaningful(diff_text):
            apply_errors.append(
                f"{rel_path}: diff rejected -- it is empty or has identical "
                "added and removed content."
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
            "blocked_reason": f"All proposed edits failed validation: {apply_errors}",
            "changed_files": [],
            "apply_errors": apply_errors,
            "model_result": result,
            "usage": usage,
            "latency_ms": latency_ms,
        }

    return {
        "status": "FILES_READY",
        "changed_files": changed_files,
        "apply_errors": apply_errors,
        "summary": result.get("summary", ""),
        "jira_key": result.get("jira_key", ""),
        "model_result": result,
        "usage": usage,
        "latency_ms": latency_ms,
    }