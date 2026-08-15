"""
agents/common/patch_apply.py (v3 — resilient + debuggable + path-safe)

CHANGE from v2: write_full_file() joined an LLM-provided relative path
straight into repo_root with no check. A path like "../../.ssh/whatever"
or an absolute path would have written outside the repo. Added a
containment check — this doesn't fully replace treating LLM output as
untrusted, but it closes the obvious traversal hole.

Everything else (multi-strategy git apply, debuggable failure messages)
unchanged from v2.
"""

import os
import subprocess
import tempfile
from typing import Tuple

DIFF_SNIPPET_MAX_CHARS = 1200

APPLY_STRATEGIES = [
    ["git", "apply", "--whitespace=fix", "--recount"],
    ["git", "apply", "--whitespace=fix", "--recount", "--unidiff-zero"],
    ["git", "apply", "--whitespace=fix", "--recount", "--ignore-space-change", "--ignore-whitespace"],
]


def _resolve_within_repo(repo_root: str, rel_path: str) -> str:
    """Resolve rel_path against repo_root and refuse anything that
    escapes it (via '..', an absolute path, or a symlink trick)."""
    repo_root_abs = os.path.realpath(repo_root)
    full_path = os.path.realpath(os.path.join(repo_root_abs, rel_path))

    if os.path.commonpath([repo_root_abs, full_path]) != repo_root_abs:
        raise ValueError(
            f"Refusing to write outside repo root: '{rel_path}' resolves to "
            f"'{full_path}', which escapes '{repo_root_abs}'."
        )
    return full_path


def write_full_file(repo_root: str, rel_path: str, content: str) -> None:
    full_path = _resolve_within_repo(repo_root, rel_path)
    os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as fh:
        fh.write(content)


def _snippet(diff_text: str) -> str:
    if len(diff_text) <= DIFF_SNIPPET_MAX_CHARS:
        return diff_text
    return diff_text[:DIFF_SNIPPET_MAX_CHARS] + "\n...[truncated]..."


def apply_unified_diff(repo_root: str, diff_text: str) -> Tuple[bool, str]:
    """Applies a unified diff via `git apply`, trying progressively more
    lenient strategies. Returns (success, message). On failure, message
    includes both git's stderr AND a snippet of the raw diff so failures
    are actually debuggable instead of just 'corrupt patch at line N'.

    Path containment for diffs is left to `git apply` itself (which
    already refuses to touch paths outside the working tree via its
    own path-prefix handling) — the explicit check above is specifically
    for write_full_file, which had no such protection at all."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".diff", delete=False, encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(diff_text)
        tmp_path = tmp_file.name

    errors = []
    try:
        for strategy in APPLY_STRATEGIES:
            result = subprocess.run(
                strategy + [tmp_path],
                cwd=repo_root, capture_output=True, text=True,
            )
            if result.returncode == 0:
                return True, f"Patch applied successfully (strategy: {' '.join(strategy)})."
            errors.append(f"[{' '.join(strategy)}] {result.stderr.strip()}")

        combined_errors = " | ".join(errors)
        return False, (
            f"git apply failed after {len(APPLY_STRATEGIES)} strategies: {combined_errors}\n"
            f"--- raw diff received from model ---\n{_snippet(diff_text)}"
        )
    finally:
        os.unlink(tmp_path)