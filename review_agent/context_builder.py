"""
review_agent/context_builder.py  (v2 — token-optimized)

Key changes from v1:
  - No more sending full file content AND the diff. The diff (with a small
    unified-context window) is the primary evidence. Full file content is
    only included for brand-new files or files under a tiny line threshold,
    where the diff already equals ~the whole file anyway.
  - Dependency context is now SYMBOL-SELECTIVE: only signatures of functions
    /classes actually referenced by identifiers in the changed lines are
    included, not every signature in every locally-imported file. This uses
    the cached repo map (repo_map.py) instead of re-parsing files every run.
  - A real token budget is enforced end-to-end, with a hard ceiling well
    below the model's per-minute limit, and priority-based degradation:
        1. diff (always kept, trimmed if needed)
        2. referenced symbol signatures (dropped first if over budget)
        3. full content of new/tiny files (dropped second)
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Set

from repo_map import RepoMap, Symbol

CHARS_PER_TOKEN_ESTIMATE = 3.3      # slightly conservative vs. 4, to avoid
                                     # underestimating real token counts
SAFETY_MARGIN = 1.15                # inflate estimates by 15% as a buffer

NEW_OR_TINY_FILE_LINE_THRESHOLD = 25
DIFF_CONTEXT_LINES = 2               # git diff -U2 instead of default -U3


@dataclass
class ContextPackage:
    diff_text: str = ""
    full_files: Dict[str, str] = field(default_factory=dict)
    referenced_signatures: Dict[str, str] = field(default_factory=dict)
    truncated: bool = False
    notes: List[str] = field(default_factory=list)
    estimated_tokens: int = 0


def estimate_tokens(text: str) -> int:
    return max(1, int((len(text) / CHARS_PER_TOKEN_ESTIMATE) * SAFETY_MARGIN))


def get_narrow_diff(repo_root: str, base_sha: str, head_sha: str) -> str:
    """Diff with a small context window — this alone usually carries enough
    signal for review without also needing the full file."""
    result = subprocess.run(
        ["git", "diff", f"-U{DIFF_CONTEXT_LINES}", base_sha, head_sha],
        cwd=repo_root, capture_output=True, text=True, check=True,
    )
    return result.stdout


def extract_identifiers(diff_text: str) -> Set[str]:
    """Pull plausible function/class/variable identifiers out of the added
    and removed diff lines, so we know which symbols to look up in the repo
    map. Deliberately simple regex — false positives just mean we look up a
    symbol that doesn't exist, which is a no-op."""
    identifiers = set()
    identifier_pattern = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")

    for line in diff_text.splitlines():
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith(("+", "-")):
            identifiers.update(identifier_pattern.findall(line[1:]))

    return identifiers


def is_new_or_tiny(repo_root: str, rel_path: str, base_sha: str) -> bool:
    """A file counts as 'new or tiny' if it didn't exist at base_sha, or it
    has few enough lines that the diff already captures ~all of it."""
    check = subprocess.run(
        ["git", "cat-file", "-e", f"{base_sha}:{rel_path}"],
        cwd=repo_root, capture_output=True,
    )
    is_new = check.returncode != 0

    full_path = os.path.join(repo_root, rel_path)
    if not os.path.exists(full_path):
        return is_new

    with open(full_path, "r", encoding="utf-8") as fh:
        line_count = sum(1 for _ in fh)

    return is_new or line_count <= NEW_OR_TINY_FILE_LINE_THRESHOLD


def build_context(
    repo_root: str,
    changed_files: List[str],
    base_sha: str,
    head_sha: str,
    repo_map: RepoMap,
    max_tokens: int = 3000,
) -> ContextPackage:
    pkg = ContextPackage()
    budget = max_tokens

    # Priority 1: the narrow diff — always kept.
    diff_text = get_narrow_diff(repo_root, base_sha, head_sha)
    diff_cost = estimate_tokens(diff_text)

    if diff_cost > budget:
        keep_chars = int(budget * CHARS_PER_TOKEN_ESTIMATE / SAFETY_MARGIN)
        diff_text = diff_text[:keep_chars]
        pkg.truncated = True
        pkg.notes.append("Diff truncated to fit token budget")
        diff_cost = estimate_tokens(diff_text)

    pkg.diff_text = diff_text
    budget -= diff_cost

    # Priority 2: symbol-selective dependency signatures (skip files already
    # in changed_files — those are covered by the diff/full content).
    if budget > 0:
        identifiers = extract_identifiers(diff_text)
        referenced: List[Symbol] = repo_map.resolve_symbols(
            identifiers, exclude_files=set(changed_files)
        )
        for symbol in referenced:
            entry = f"{symbol.file}: {symbol.signature}"
            cost = estimate_tokens(entry)
            if cost > budget:
                pkg.notes.append(f"Dropped symbol {symbol.name} (budget exceeded)")
                pkg.truncated = True
                continue
            pkg.referenced_signatures[f"{symbol.file}:{symbol.name}"] = entry
            budget -= cost

    # Priority 3: full content of new/tiny changed files only.
    for rel_path in changed_files:
        if budget <= 0:
            pkg.notes.append(f"{rel_path} full content dropped (out of budget)")
            pkg.truncated = True
            continue
        if not is_new_or_tiny(repo_root, rel_path, base_sha):
            continue
        full_path = os.path.join(repo_root, rel_path)
        if not os.path.exists(full_path):
            continue
        with open(full_path, "r", encoding="utf-8") as fh:
            content = fh.read()
        cost = estimate_tokens(content)
        if cost <= budget:
            pkg.full_files[rel_path] = content
            budget -= cost
        else:
            pkg.notes.append(f"{rel_path} full content dropped (exceeds remaining budget)")
            pkg.truncated = True

    pkg.estimated_tokens = max_tokens - budget
    return pkg


def render_context_for_prompt(pkg: ContextPackage) -> str:
    parts = [f"## Diff\n```diff\n{pkg.diff_text}\n```"]

    if pkg.full_files:
        parts.append("## New / small files (full content)")
        for path, content in pkg.full_files.items():
            parts.append(f"### {path}\n```python\n{content}\n```")

    if pkg.referenced_signatures:
        parts.append("## Referenced symbols (signature only)")
        for key, entry in pkg.referenced_signatures.items():
            parts.append(f"- {entry}")

    if pkg.truncated:
        parts.append("## Context notes\n" + "\n".join(f"- {n}" for n in pkg.notes))

    return "\n\n".join(parts)