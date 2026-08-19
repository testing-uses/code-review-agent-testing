"""agents/dev_agent/dev_agent.py  (v6)

CHANGES from v5 (the "strict" version):
  - Adapted to the new array-based new_files/diffs wire format (see
    groq_client.DEV_AGENT_RESPONSE_SCHEMA for why) -- _validate_edit_contract
    now checks arrays of {path, content|diff} objects, and a new
    _edits_to_dicts() converts them back into the plain {path: content}
    dicts the apply logic already worked with, so nothing below that
    point had to change.
  - max_output_tokens is no longer a hardcoded 1500. Ground-truth file
    injection was never counted against the DCBA-allocated budget at
    all (up to ~7,200+ tokens uncapped, competing with a fixed 1500
    output limit for the same ceiling) -- under strict-schema constrained
    decoding, running out of room mid-generation doesn't degrade to
    truncated-but-parseable JSON, it just fails outright with an empty
    failed_generation. _compute_dynamic_budgets() now splits the actual
    DCBA ceiling three ways (output / KB context / ground truth) so the
    budget DCBA computed is the budget that's actually enforced, not
    decorative.
  - _build_ground_truth_block() now takes a real token budget and
    truncates the AGGREGATE ground-truth text to fit it, instead of a
    flat 6000-chars-per-file cap with no total limit.

Everything else (ground-truth enforcement at apply time, no-op diff
rejection, rewrite-similarity guard, protected-path checks) is unchanged
from v5 and still applies to the converted dict form.
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
    CHARS_PER_TOKEN_ESTIMATE,
    DEV_AGENT_RESPONSE_SCHEMA,
    GroqKeyPool,
    call_groq_json,
    load_prompt,
)
from kb_schema import get_connection  # noqa: E402
from patch_apply import apply_unified_diff, write_full_file  # noqa: E402

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
DEFAULT_MODEL = "openai/gpt-oss-120b"

GROUND_TRUTH_MAX_FILES = 4
REWRITE_SIMILARITY_FLOOR = 0.5

# Dynamic budget split (see _compute_dynamic_budgets docstring).
MIN_OUTPUT_TOKENS = 900
MAX_OUTPUT_TOKENS_CAP = 4000
OUTPUT_BUDGET_FRACTION = 0.35
KB_CONTEXT_FRACTION_OF_REMAINDER = 0.35

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


def _compute_dynamic_budgets(allocated_budget_tokens: int) -> Tuple[int, int, int]:
    """Split the DCBA-allocated ceiling three ways instead of the old
    fixed 0.6-for-KB-context / hardcoded-1500-for-output split that let
    ground truth injection silently eat the whole budget.

    Returns (output_budget_tokens, kb_context_budget_tokens, ground_truth_budget_tokens).
    All three plus the system+user prompt overhead stay within
    allocated_budget_tokens by construction, so preflight_check should
    never reject a request assembled from these numbers under normal
    conditions."""
    output_budget = max(
        MIN_OUTPUT_TOKENS,
        min(MAX_OUTPUT_TOKENS_CAP, int(allocated_budget_tokens * OUTPUT_BUDGET_FRACTION)),
    )
    remainder = max(allocated_budget_tokens - output_budget, 400)
    kb_budget = int(remainder * KB_CONTEXT_FRACTION_OF_REMAINDER)
    ground_truth_budget = remainder - kb_budget
    return output_budget, kb_budget, ground_truth_budget


def _read_existing_file(
    repo_root: str,
    rel_path: str,
    max_chars: Optional[int] = None,
) -> Optional[str]:
    full_path = os.path.join(repo_root, rel_path)

    if not os.path.isfile(full_path):
        return None

    try:
        with open(full_path, "r", encoding="utf-8") as file_handle:
            content = file_handle.read()
    except (OSError, UnicodeDecodeError):
        return None

    if max_chars is not None and len(content) > max_chars:
        content = (
            content[:max_chars]
            + "\n...[truncated -- file longer than ground-truth budget]..."
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
    budget_tokens: int,
) -> Tuple[str, Set[str]]:
    """Truncates the AGGREGATE ground-truth block to fit budget_tokens
    (not a flat per-file cap) -- so a task naming several files degrades
    gracefully (later/larger files get less room, or get dropped) instead
    of silently producing a prompt that blows past the DCBA ceiling."""
    max_total_chars = max(int(budget_tokens * CHARS_PER_TOKEN_ESTIMATE), 500)
    blocks: List[str] = []
    verified: Set[str] = set()
    used_chars = 0

    for rel_path in target_files:
        remaining_chars = max_total_chars - used_chars
        if remaining_chars <= 200:
            break  # not enough budget left for another file to be useful

        content = _read_existing_file(repo_root, rel_path, max_chars=remaining_chars)
        if content is None:
            continue

        verified.add(rel_path)
        block = f"### EXACT CURRENT CONTENT of {rel_path}\n```python\n{content}\n```"
        blocks.append(block)
        used_chars += len(block)

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
    """Validates the array-based wire format: new_files/diffs are arrays
    of {path, content} / {path, diff} objects (see groq_client's schema
    docstring for why -- strict mode can't express a dict keyed by
    arbitrary file paths)."""
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

    for field, item_key in (("new_files", "content"), ("diffs", "diff")):
        value = result[field]
        if not isinstance(value, list):
            return False, f"{field} must be an array"

        seen_paths: Set[str] = set()
        for entry in value:
            if not isinstance(entry, dict):
                return False, f"{field} entries must be objects"
            if set(entry.keys()) != {"path", item_key}:
                return False, f"{field} entries must have exactly 'path' and '{item_key}'"
            if not isinstance(entry["path"], str) or not isinstance(entry[item_key], str):
                return False, f"{field} entries must have string 'path' and '{item_key}'"
            if entry["path"] in seen_paths:
                return False, f"{field} contains duplicate path: {entry['path']}"
            seen_paths.add(entry["path"])

    new_file_paths = {entry["path"] for entry in result["new_files"]}
    diff_paths = {entry["path"] for entry in result["diffs"]}
    overlap = new_file_paths & diff_paths
    if overlap:
        return False, f"paths appear in both new_files and diffs: {sorted(overlap)}"

    if result["blocked"]:
        if result["new_files"] or result["diffs"]:
            return False, "blocked responses must contain empty edit arrays"
    elif not result["new_files"] and not result["diffs"]:
        return False, "unblocked responses must contain at least one edit"

    return True, ""


def _edits_to_dicts(result: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Converts the wire format (arrays of {path, content|diff}) into the
    {path: content} dicts the apply logic below expects -- keeps that
    logic identical to before the schema had to switch to arrays."""
    new_files = {
        entry["path"].replace("\\", "/"): entry["content"]
        for entry in result["new_files"]
    }
    diffs = {
        entry["path"].replace("\\", "/"): entry["diff"]
        for entry in result["diffs"]
    }
    return new_files, diffs


def run(
    task_text: str,
    repo_root: str,
    db_path: str,
    allocated_budget_tokens: int,
    model: str = DEFAULT_MODEL,
    max_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    start_time = time.time()
    system_prompt = load_prompt(PROMPTS_DIR, "dev_agent_prompt.md")

    computed_output_budget, context_budget, ground_truth_budget = _compute_dynamic_budgets(
        allocated_budget_tokens
    )
    if max_output_tokens is None:
        max_output_tokens = computed_output_budget

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
        budget_tokens=ground_truth_budget,
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
        "jira_key, new_files, diffs. new_files and diffs are JSON ARRAYS; "
        "each element is an object with exactly two string fields -- "
        "{\"path\": ..., \"content\": ...} for new_files, "
        "{\"path\": ..., \"diff\": ...} for diffs. Never repeat the same "
        "path twice within one array, and never put the same path in both "
        "arrays. Never return arrays of lines, nested objects, null, "
        "numbers, Markdown, or prose. For an existing ground-truth file, "
        "return the complete updated file as one string in a new_files "
        "entry. If blocked is true, both arrays must be empty."
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

    new_files, diffs = _edits_to_dicts(result)
    changed_files: List[str] = []
    apply_errors: List[str] = []

    for rel_path, content in new_files.items():
        if rel_path.startswith("agents/") or rel_path.startswith(".github/"):
            apply_errors.append(
                f"{rel_path}: protected platform path cannot be modified"
            )
            continue

        existing_content = _read_existing_file(repo_root, rel_path)

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