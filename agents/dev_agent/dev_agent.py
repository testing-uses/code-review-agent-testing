"""agents/dev_agent/dev_agent.py

Dev Agent with optional Cerebras provider support.

Existing Groq behavior remains available. Set DEV_AGENT_PROVIDER=cerebras
and CEREBRAS_API_KEY to use Cerebras for the Dev Agent only. The rest of the
workflow is unchanged.
"""

from __future__ import annotations

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

from context_selector import select_context  # noqa: E402
from groq_client import (  # noqa: E402
    CHARS_PER_TOKEN_ESTIMATE,
    DEV_AGENT_RESPONSE_SCHEMA,
    GroqKeyPool,
    call_cerebras_json,
    call_groq_json,
    estimate_tokens,
    load_prompt,
)
from kb_schema import get_connection  # noqa: E402
from patch_apply import apply_unified_diff, write_full_file  # noqa: E402

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
DEFAULT_MODEL = os.environ.get(
    "DEV_AGENT_MODEL",
    "openai/gpt-oss-120b",
)
DEFAULT_PROVIDER = os.environ.get(
    "DEV_AGENT_PROVIDER",
    "groq",
).lower()

GROUND_TRUTH_MAX_FILES = 4
REWRITE_SIMILARITY_FLOOR = 0.5
PRELIM_OUTPUT_FRACTION = 0.35
PRELIM_KB_FRACTION_OF_REMAINDER = 0.35
MIN_OUTPUT_TOKENS = 2500
MAX_OUTPUT_TOKENS_CAP = 8000
PROMPT_SAFETY_MARGIN_TOKENS = 200

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


def _compute_preliminary_budgets(
    allocated_budget_tokens: int,
) -> Tuple[int, int, int]:
    prelim_output = max(
        MIN_OUTPUT_TOKENS,
        min(
            MAX_OUTPUT_TOKENS_CAP,
            int(allocated_budget_tokens * PRELIM_OUTPUT_FRACTION),
        ),
    )
    remainder = max(allocated_budget_tokens - prelim_output, 400)
    kb_budget = int(remainder * PRELIM_KB_FRACTION_OF_REMAINDER)
    ground_truth_budget = remainder - kb_budget
    return prelim_output, kb_budget, ground_truth_budget


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
    max_total_chars = max(
        int(budget_tokens * CHARS_PER_TOKEN_ESTIMATE),
        500,
    )
    blocks: List[str] = []
    verified: Set[str] = set()
    used_chars = 0

    for rel_path in target_files:
        remaining_chars = max_total_chars - used_chars
        if remaining_chars <= 200:
            break

        content = _read_existing_file(
            repo_root,
            rel_path,
            max_chars=remaining_chars,
        )
        if content is None:
            continue

        verified.add(rel_path)
        block = (
            f"### EXACT CURRENT CONTENT of {rel_path}\n"
            f"```python\n{content}\n```"
        )
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

    for field, item_key in (
        ("new_files", "content"),
        ("diffs", "diff"),
    ):
        value = result[field]
        if not isinstance(value, list):
            return False, f"{field} must be an array"

        seen_paths: Set[str] = set()
        for entry in value:
            if not isinstance(entry, dict):
                return False, f"{field} entries must be objects"
            if set(entry.keys()) != {"path", item_key}:
                return False, (
                    f"{field} entries must have exactly 'path' and "
                    f"'{item_key}'"
                )
            if not isinstance(entry["path"], str):
                return False, f"{field} path must be a string"
            if not isinstance(entry[item_key], str):
                return False, f"{field} {item_key} must be a string"
            if entry["path"] in seen_paths:
                return False, (
                    f"{field} contains duplicate path: "
                    f"{entry['path']}"
                )
            seen_paths.add(entry["path"])

    new_file_paths = {
        entry["path"] for entry in result["new_files"]
    }
    diff_paths = {entry["path"] for entry in result["diffs"]}
    overlap = new_file_paths & diff_paths
    if overlap:
        return False, (
            f"paths appear in both new_files and diffs: "
            f"{sorted(overlap)}"
        )

    if result["blocked"]:
        if result["new_files"] or result["diffs"]:
            return False, "blocked responses must contain empty edit arrays"
    elif not result["new_files"] and not result["diffs"]:
        return False, "unblocked responses must contain at least one edit"

    return True, ""


def _edits_to_dicts(
    result: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    new_files = {
        entry["path"].replace("\\", "/"): entry["content"]
        for entry in result["new_files"]
    }
    diffs = {
        entry["path"].replace("\\", "/"): entry["diff"]
        for entry in result["diffs"]
    }
    return new_files, diffs


def _call_provider(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    allocated_budget_tokens: int,
) -> Dict[str, Any]:
    if provider == "cerebras":
        return call_cerebras_json(
            model=os.getenv(
                "CEREBRAS_MODEL",
                model,
            ),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
        )

    key_pool = GroqKeyPool()
    return call_groq_json(
        key_pool=key_pool,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_output_tokens=max_output_tokens,
        token_ceiling=allocated_budget_tokens,
        response_schema=DEV_AGENT_RESPONSE_SCHEMA,
    )


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

    _, prelim_kb_budget, prelim_ground_truth_budget = (
        _compute_preliminary_budgets(allocated_budget_tokens)
    )

    if os.path.exists(db_path):
        connection = get_connection(db_path)
        try:
            kb_context = select_context(
                connection,
                task_text,
                budget_tokens=prelim_kb_budget,
            )
        finally:
            connection.close()
    else:
        kb_context = "(knowledge base not yet built)"

    target_files = _guess_target_files(task_text, repo_root)
    ground_truth_block, verified_ground_truth_files = (
        _build_ground_truth_block(
            repo_root,
            target_files,
            budget_tokens=prelim_ground_truth_budget,
        )
    )

    user_prompt = (
        f"## Task\n{task_text}\n\n"
        "## Retrieved context\n"
        "This context is supplementary and is not ground truth.\n"
        f"{kb_context}\n\n"
        "## Exact current file content -- authoritative ground truth\n"
        "Make only the requested change. Preserve all unrelated content. "
        "If the required existing file is not shown, set blocked=true. "
        "For a small file or a change touching most of the file, return "
        "complete updated content. For a larger file with a localized "
        "change, return a unified diff against the exact content shown.\n\n"
        f"{ground_truth_block}\n\n"
        "## Non-negotiable output contract\n"
        "Return exactly one JSON object and nothing else. It must contain "
        "blocked, blocked_reason, summary, jira_key, new_files, diffs. "
        "new_files and diffs are arrays. Each new_files item has exactly "
        "path and content string fields. Each diffs item has exactly path "
        "and diff string fields. Never return arrays of source lines, "
        "nested objects, null, numbers, Markdown, or prose. Do not repeat "
        "a path. A blocked response must have both arrays empty."
    )

    prompt_tokens_actual = (
        estimate_tokens(system_prompt)
        + estimate_tokens(user_prompt)
    )

    if max_output_tokens is None:
        reclaimed_output = (
            allocated_budget_tokens
            - prompt_tokens_actual
            - PROMPT_SAFETY_MARGIN_TOKENS
        )

        if reclaimed_output < MIN_OUTPUT_TOKENS:
            return {
                "status": "BLOCKED",
                "blocked_reason": (
                    f"Allocated budget ({allocated_budget_tokens}) leaves "
                    f"only approximately {max(reclaimed_output, 0)} output "
                    f"tokens after the estimated prompt ({prompt_tokens_actual})."
                ),
                "usage": {},
                "latency_ms": (time.time() - start_time) * 1000,
            }

        max_output_tokens = min(
            MAX_OUTPUT_TOKENS_CAP,
            reclaimed_output,
        )

    provider = os.getenv(
        "DEV_AGENT_PROVIDER",
        DEFAULT_PROVIDER,
    ).lower()

    print(
        "[DEV_AGENT_TOKEN_DEBUG] "
        + json.dumps(
            {
                "provider": provider,
                "model": model,
                "allocated_budget_tokens": allocated_budget_tokens,
                "system_prompt_tokens": estimate_tokens(system_prompt),
                "user_prompt_tokens": estimate_tokens(user_prompt),
                "prompt_tokens": prompt_tokens_actual,
                "max_output_tokens": max_output_tokens,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    result = _call_provider(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_output_tokens=max_output_tokens,
        allocated_budget_tokens=allocated_budget_tokens,
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

        existing_content = _read_existing_file(
            repo_root,
            rel_path,
        )

        if (
            existing_content is not None
            and rel_path not in verified_ground_truth_files
        ):
            apply_errors.append(
                f"{rel_path}: exact ground truth was not supplied"
            )
            continue

        if existing_content is not None and not _validate_preserves_structure(
            existing_content,
            content,
        ):
            apply_errors.append(
                f"{rel_path}: rewrite is too dissimilar to existing file"
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

        existing_content = _read_existing_file(repo_root, rel_path)
        if (
            existing_content is not None
            and rel_path not in verified_ground_truth_files
        ):
            apply_errors.append(
                f"{rel_path}: exact ground truth was not supplied"
            )
            continue

        if not _diff_is_meaningful(diff_text):
            apply_errors.append(f"{rel_path}: diff is empty or a no-op")
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