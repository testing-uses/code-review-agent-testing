"""
review_agent/context_builder.py  (v3 — unified with the central KB)

CHANGE from v2: symbol-selective dependency context used to come from
repo_map.py, a second, disconnected symbol cache that only the review
agent knew about. That's the opposite of "one central knowledge base
shared across all contributors" — the review agent couldn't see anything
the dev agent's KB had already indexed, and the two caches could drift
out of sync independently.

Now this pulls referenced-symbol signatures straight from the same
kb.sqlite3 the Dev Agent and context_selector use (via kb_query), so
there's exactly one graph/index in the system, not two. repo_map.py is
no longer needed by this module.

Everything else is unchanged: the diff is still the primary evidence,
full file content is still limited to new/tiny files, and a hard token
budget with priority-based degradation still applies:
    1. diff (always kept, trimmed if needed)
    2. referenced symbol signatures, resolved from the central KB
    3. full content of new/tiny files (dropped first if over budget)
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from kb_schema import get_connection
from kb_query import get_signatures_excluding

CHARS_PER_TOKEN_ESTIMATE = 3.3      # slightly conservative vs. 4, to avoid
                                     # underestimating real token counts
SAFETY_MARGIN = 1.15                # inflate estimates by 15% as a buffer

NEW_OR_TINY_FILE_LINE_THRESHOLD = 25
DIFF_CONTEXT_LINES = 2               # git diff -U2 instead of default -U3


import ast


def get_file_at_ref(repo_root: str, ref: str, rel_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{rel_path}"],
        cwd=repo_root, capture_output=True, text=True,
    )
    return result.stdout if result.returncode == 0 else ""


def extract_top_level_names(source: str) -> Set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def find_removed_symbols(repo_root: str, rel_path: str, base_sha: str) -> Set[str]:
    """Symbols defined at base_sha that no longer exist (or are now commented
    out / deleted) in the current working tree version of the file."""
    old_source = get_file_at_ref(repo_root, base_sha, rel_path)
    full_path = os.path.join(repo_root, rel_path)
    new_source = ""
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as fh:
            new_source = fh.read()

    old_names = extract_top_level_names(old_source)
    new_names = extract_top_level_names(new_source)
    return old_names - new_names


def find_downstream_usages(repo_root: str, symbol_name: str, exclude_file: str) -> List[Tuple[str, str, str]]:
    """Find other .py files in the repo that still reference a symbol that
    was just removed from the changed file. This is the reverse-dependency
    check the diff alone can never surface."""
    result = subprocess.run(
        ["git", "grep", "-n", "-w", symbol_name, "--", "*.py"],
        cwd=repo_root, capture_output=True, text=True,
    )
    usages = []
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        path, line_number, content = parts
        if path == exclude_file:
            continue
        usages.append((path, line_number, content.strip()))
    return usages


@dataclass
class ContextPackage:
    diff_text: str = ""
    full_files: Dict[str, str] = field(default_factory=dict)
    referenced_signatures: Dict[str, str] = field(default_factory=dict)
    removed_symbols_with_usages: List[Tuple[str, str, List[Tuple[str, str, str]]]] = field(default_factory=list)
    truncated: bool = False
    notes: List[str] = field(default_factory=list)
    estimated_tokens: int = 0


def estimate_tokens(text: str) -> int:
    return max(1, int((len(text) / CHARS_PER_TOKEN_ESTIMATE) * SAFETY_MARGIN))


def get_narrow_diff(
    repo_root: str,
    base_sha: str,
    head_sha: str,
    changed_files: List[str],
) -> str:
    command = [
        "git",
        "diff",
        f"-U{DIFF_CONTEXT_LINES}",
        base_sha,
        head_sha,
        "--",
        *changed_files,
    ]

    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout


def extract_identifiers(diff_text: str) -> Set[str]:
    """Pull plausible function/class/variable identifiers out of the added
    and removed diff lines, so we know which symbols to look up in the
    central KB. Deliberately simple regex — false positives just mean we
    look up a symbol that doesn't exist, which is a no-op."""
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
    db_path: str,
    max_tokens: int = 3000,
) -> ContextPackage:
    """db_path points at the SAME kb.sqlite3 the Dev Agent / context_selector
    use — this is the "current repo snapshot" the review agent gets: not
    the whole codebase, just the diff plus the handful of KB-indexed
    signatures actually referenced by the changed lines."""
    pkg = ContextPackage()
    budget = max_tokens

    # Priority 1: the narrow diff — always kept.
    diff_text = get_narrow_diff(
        repo_root,
        base_sha,
        head_sha,
        changed_files,
    )
    diff_cost = estimate_tokens(diff_text)

    removed_symbols_report = []
    for rel_path in changed_files:
        if not rel_path.endswith(".py"):
            continue
        removed = find_removed_symbols(repo_root, rel_path, base_sha)
        for symbol in removed:
            usages = find_downstream_usages(repo_root, symbol, rel_path)
            if usages:
                removed_symbols_report.append((symbol, rel_path, usages))

    pkg.removed_symbols_with_usages = removed_symbols_report

    if diff_cost > budget:
        keep_chars = int(budget * CHARS_PER_TOKEN_ESTIMATE / SAFETY_MARGIN)
        diff_text = diff_text[:keep_chars]
        pkg.truncated = True
        pkg.notes.append("Diff truncated to fit token budget")
        diff_cost = estimate_tokens(diff_text)

    pkg.diff_text = diff_text
    budget -= diff_cost

    # Priority 2: symbol-selective dependency signatures, resolved from the
    # CENTRAL knowledge base (skip files already in changed_files — those
    # are covered by the diff/full content already).
    if budget > 0 and os.path.exists(db_path):
        identifiers = extract_identifiers(diff_text)
        conn = get_connection(db_path)
        try:
            referenced = get_signatures_excluding(
                conn, identifiers, exclude_files=set(changed_files)
            )
        finally:
            conn.close()

        for symbol in referenced:
            entry = f"{symbol['file']}: {symbol['signature']}"
            cost = estimate_tokens(entry)
            if cost > budget:
                pkg.notes.append(f"Dropped symbol {symbol['name']} (budget exceeded)")
                pkg.truncated = True
                continue
            pkg.referenced_signatures[f"{symbol['file']}:{symbol['name']}"] = entry
            budget -= cost
    elif budget > 0:
        pkg.notes.append(f"Central KB not found at {db_path} — skipping symbol context. Run build_kb.py first.")

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

    if pkg.removed_symbols_with_usages:
        parts.append("## ⚠ BREAKING CHANGE WARNING — removed symbols still referenced elsewhere")
        for symbol, source_file, usages in pkg.removed_symbols_with_usages:
            parts.append(f"`{symbol}` was removed from `{source_file}` but is still used in:")
            for path, line_number, content in usages:
                parts.append(f"  - {path}:{line_number}: `{content}`")

    if pkg.referenced_signatures:
        parts.append("## Referenced symbols (signature only, from central KB)")
        for key, entry in pkg.referenced_signatures.items():
            parts.append(f"- {entry}")

    if pkg.truncated:
        parts.append("## Context notes\n" + "\n".join(f"- {n}" for n in pkg.notes))

    return "\n\n".join(parts)