"""
agents/common/patch_apply.py (v2 — resilient + debuggable)

Changes from v1:
- Tries multiple `git apply` strategies before giving up, because LLM-
  generated unified diffs frequently have correct content but slightly
  wrong hunk-header line counts (`@@ -a,b +c,d @@`). `--recount` tells
  git to ignore the header counts and recompute them from the actual
  hunk body, which fixes the single most common LLM diff failure mode.
- On total failure, the raw diff text (truncated) is included in the
  returned message, so the orchestrator's BLOCKED reason actually shows
  what the model produced instead of just git's opaque stderr.
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


def write_full_file(repo_root: str, rel_path: str, content: str) -> None:
    full_path = os.path.join(repo_root, rel_path)
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
    are actually debuggable instead of just 'corrupt patch at line N'."""
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