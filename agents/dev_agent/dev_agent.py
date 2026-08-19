"""agents/dev_agent/dev_agent.py

Strict Dev Agent implementation.

The model output contract is enforced at three levels:
1. The system prompt requires one exact JSON shape.
2. Groq receives a strict JSON Schema.
3. Python validates the parsed response before applying anything.

Existing files named in the task are supplied as exact ground truth and are
returned as complete strings under new_files. Diffs are rejected for those
files. No model-generated arrays, nested objects, or alternate file formats
are accepted.
"""

from __future__ import annotations

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

from context_selector import select_context  # noqa: E402
from groq_client import (  # noqa: E402
    DEV_AGENT_RESPONSE_SCHEMA,
    GroqKeyPool,
    call_groq_json,
    load_prompt,
)
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
    blocks: List[str] = []
    verified: Set[str] = set()

    for rel_path in target_files:
        content = _read_existing_file(repo_root, rel_path)
        if content is None:
            continue

        verified.add(rel_path)
        blocks.append(
            f"### EXACT CURRENT CONTENT of {rel_path}\n"
            f"```python\n{content}\n```"
        )

    if not blocks:
        return (
            "(no exact file content available -- set blocked=true; "
            "do not guess)",
            verified,
        )

    return "\n\n".join(blocks), verified


def _validate_preserves_structure(
    original: str,
    rewritten: str,
    min_similarity: float = REWRITE_SIMILARITY_FLOOR,
) -> bool:
    if not original:
        return True

    ratio = difflib.SequenceMatcher(None, original, rewritten).ratio()
    return ratio >= min_similarity


def _diff_is_meaningful(diff_text: str) -> bool:
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

    return Counter(removed) != Counter(added)


def _validate_edit_contract(result: Any) -> Tuple[bool, str]:
    if not isinstance(result, dict):
        return False, "response must be a JSON object"

    required = {
        "blocked",
        "blocked_reason",
        "summary",
        "jira_key",
        "diffs",
        "new_files",
    }
    missing = required - result.keys()
    if missing:
        return False, f"missing fields: {sorted(missing)}"

    if not isinstance(result["blocked"], bool):
        return False, "blocked must be a boolean"

    for field in ("blocked_reason", "summary", "jira_key"):
        if not isinstance(result[field], str):
            return False, f"{field} must be a string"

    for field in ("diffs", "new_files"):
        value = result[field]
        if not isinstance(value, dict):
            return False, f"{field} must be an object"

        for path, content in value.items():
            if not isinstance(path, str):
                return False, f"{field} contains a non-string path"
            if not isinstance(content, str):
                return False, (
                    f"{field}[{path!r}] must be one complete string; "
                    "arrays and nested objects are forbidden"
                )

    overlap = set(result["diffs"]) & set(result["new_files"])
    if overlap:
        return False, f"paths appear in both output fields: {sorted(overlap)}"

    if result["blocked"]:
        if result["diffs"] or result["new_files"]:
            return False, "blocked responses must contain empty edit objects"
    elif not result["diffs"] and not result["new_files"]:
        return False, "unblocked responses must contain at least one edit"

    return True, ""


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
        kb_context = "(knowledge base not yet built)"

    target_files = _guess_target_files(task_text, repo_root)
    ground_truth_block, verified_ground_truth_files = _build_ground_truth_block(
        repo_root,
        target_files,
    )

    user_prompt = (
        f"## Task\n{task_text}\n\n"
        "## Retrieved context\n"
        "This context is supplementary and is not ground truth.\n"
        f"{kb_context}\n\n"
        "## Exact current file content -- authoritative ground truth\n"
        "Make only the requested change. Preserve all unrelated content. "
        "If the required existing file is not shown, set blocked=true.\n\n"
        f"{ground_truth_block}\n\n"
        "## Non-negotiable output contract\n"
        "Return exactly one JSON object and nothing else. The object must "
        "contain exactly these fields: blocked, blocked_reason, summary, "
        "jira_key, diffs, new_files. The values of diffs and new_files "
        "must be JSON objects whose values are single complete strings. "
        "Never return arrays of lines, nested objects, null, numbers, "
        "Markdown, or prose. For an existing ground-truth file, return "
        "the complete updated file as one string in new_files. If blocked "
        "is true, both edit objects must be empty."
    )

    key_pool = GroqKeyPool()
    result = call_groq_json(
        key_pool=key_pool,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_output_tokens=max_output_tokens,
        token_ceiling=allocated_budget_tokens,
        response_schema=DEV_AGENT_RESPONSE_SCHEMA,
    )

    usage = result.pop("_usage", {})
    latency_ms = (time.time() - start_time) * 1000

    valid, contract_error = _validate_edit_contract(result)
    if not valid:
        return {
            "status": "BLOCKED",
            "blocked_reason": (
                "Model violated the Dev Agent response contract: "
                f"{contract_error}"
            ),
            "model_result": result,
            "usage": usage,
            "latency_ms": latency_ms,
        }

    if result["blocked"]:
        return {
            "status": "BLOCKED",
            "blocked_reason": result["blocked_reason"],
            "model_result": result,
            "usage": usage,
            "latency_ms": latency_ms,
        }

    diffs = result["diffs"]
    new_files = result["new_files"]
    changed_files: List[str] = []
    apply_errors: List[str] = []

    for rel_path, content in new_files.items():
        rel_path = rel_path.replace("\\", "/")

        if rel_path.startswith("agents/") or rel_path.startswith(".github/"):
            apply_errors.append(
                f"{rel_path}: protected platform path cannot be modified"
            )
            continue

        existing_content = _read_existing_file(
            repo_root,
            rel_path,
            max_chars=1_000_000,
        )

        if (
            existing_content is not None
            and rel_path not in verified_ground_truth_files
        ):
            apply_errors.append(
                f"{rel_path}: refusing overwrite because exact ground truth "
                "was not supplied to the model"
            )
            continue

        if existing_content is not None and not _validate_preserves_structure(
            existing_content,
            content,
        ):
            apply_errors.append(
                f"{rel_path}: rewrite is too dissimilar to the existing file"
            )
            continue

        try:
            write_full_file(repo_root, rel_path, content)
            changed_files.append(rel_path)
        except (OSError, ValueError) as error:
            apply_errors.append(f"{rel_path}: {error}")

    for rel_path, diff_text in diffs.items():
        rel_path = rel_path.replace("\\", "/")

        if rel_path.startswith("agents/") or rel_path.startswith(".github/"):
            apply_errors.append(
                f"{rel_path}: protected platform path cannot be modified"
            )
            continue

        if rel_path in verified_ground_truth_files:
            apply_errors.append(
                f"{rel_path}: diffs are forbidden for exact ground-truth files"
            )
            continue

        if not _diff_is_meaningful(diff_text):
            apply_errors.append(
                f"{rel_path}: diff is empty or a no-op"
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
            "blocked_reason": (
                f"All proposed edits failed validation: {apply_errors}"
            ),
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
        "summary": result["summary"],
        "jira_key": result["jira_key"],
        "model_result": result,
        "usage": usage,
        "latency_ms": latency_ms,
    }