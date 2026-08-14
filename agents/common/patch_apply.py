"""
agents/common/patch_apply.py

The deterministic mechanism the Dev Agent uses to "edit" a file. The LLM
never executes anything itself — it only produces text, which this module
applies:

    New file       -> write_full_file(): plain file write.
    Existing file  -> apply_unified_diff(): `git apply` on an LLM-produced
                       unified diff. This fails loudly (non-zero exit,
                       captured stderr) instead of silently corrupting the
                       file if the diff doesn't cleanly match — which is
                       exactly the auditability you want from an agent that
                       edits code unsupervised.
"""

import os
import subprocess
import tempfile
from typing import Tuple


def write_full_file(repo_root: str, rel_path: str, content: str) -> None:
    full_path = os.path.join(repo_root, rel_path)
    os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as fh:
        fh.write(content)


def apply_unified_diff(repo_root: str, diff_text: str) -> Tuple[bool, str]:
    """Applies a unified diff via `git apply`. Returns (success, message)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".diff", delete=False, encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(diff_text)
        tmp_path = tmp_file.name

    try:
        result = subprocess.run(
            ["git", "apply", "--whitespace=fix", tmp_path],
            cwd=repo_root, capture_output=True, text=True,
        )
        if result.returncode == 0:
            return True, "Patch applied successfully."
        return False, f"git apply failed: {result.stderr.strip()}"
    finally:
        os.unlink(tmp_path)
