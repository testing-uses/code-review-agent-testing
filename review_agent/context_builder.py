"""
review_agent/context_builder.py

Builds a token-budgeted, dependency-aware context package for the LLM.

Strategy (priority order when the token budget is exceeded):
  1. Full content of files actually changed in the PR.
  2. Signature-only view (function/class signatures + docstrings) of files
     that import, or are imported by, the changed files.
  3. Diff-only fallback if even signatures don't fit.

For files that are individually too large to fit even alone, the file is
chunked by function/class boundaries and reviewed via map-reduce
(see reviewer.py: review_large_file_chunks).
"""

import ast
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List

CHARS_PER_TOKEN_ESTIMATE = 4  # rough heuristic, good enough for budgeting


@dataclass
class ContextPackage:
    changed_files: Dict[str, str] = field(default_factory=dict)
    dependency_signatures: Dict[str, str] = field(default_factory=dict)
    diff_text: str = ""
    truncated: bool = False
    notes: List[str] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def extract_signatures(source: str) -> str:
    """Return a signature-only view: function/class defs + docstrings, no bodies."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "# (unparsable file, signatures unavailable)"

    lines = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ", ".join(a.arg for a in node.args.args)
            lines.append(f"def {node.name}({args}): ...")
            doc = ast.get_docstring(node)
            if doc:
                lines.append(f'    """{doc.strip()}"""')
        elif isinstance(node, ast.ClassDef):
            lines.append(f"class {node.name}:")
            doc = ast.get_docstring(node)
            if doc:
                lines.append(f'    """{doc.strip()}"""')
    return "\n".join(lines) if lines else "# (no top-level defs found)"


def find_local_dependencies(changed_files: List[str], repo_root: str) -> List[str]:
    """
    Very small dependency resolver for a flat-module POC repo:
    looks for `import X` / `from X import` referencing other local .py files.
    Replace with a proper import graph (e.g. via `modulegraph`) for larger repos.
    """
    local_modules = {
        os.path.splitext(f)[0]
        for f in os.listdir(repo_root)
        if f.endswith(".py")
    }
    deps = set()
    import_re = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z_][\w]*)")

    for changed in changed_files:
        path = os.path.join(repo_root, changed)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                match = import_re.match(line)
                if match:
                    mod = match.group(1)
                    if mod in local_modules and f"{mod}.py" not in changed_files:
                        deps.add(f"{mod}.py")
    return sorted(deps)


def build_context(
    repo_root: str,
    changed_files: List[str],
    diff_text: str,
    max_tokens: int = 6000,
) -> ContextPackage:
    pkg = ContextPackage(diff_text=diff_text)
    budget = max_tokens

    # Priority 1: full content of changed files
    for rel_path in changed_files:
        full_path = os.path.join(repo_root, rel_path)
        if not os.path.exists(full_path):
            continue
        with open(full_path, "r", encoding="utf-8") as fh:
            content = fh.read()
        cost = estimate_tokens(content)
        if cost <= budget:
            pkg.changed_files[rel_path] = content
            budget -= cost
        else:
            pkg.changed_files[rel_path] = content[: budget * CHARS_PER_TOKEN_ESTIMATE]
            pkg.truncated = True
            pkg.notes.append(f"{rel_path} truncated to fit token budget")
            budget = 0
            break

    # Priority 2: signature-only view of local dependencies
    dep_files = find_local_dependencies(changed_files, repo_root)
    for rel_path in dep_files:
        if budget <= 0:
            pkg.notes.append(f"{rel_path} dropped (out of budget)")
            pkg.truncated = True
            continue
        full_path = os.path.join(repo_root, rel_path)
        with open(full_path, "r", encoding="utf-8") as fh:
            content = fh.read()
        sig = extract_signatures(content)
        cost = estimate_tokens(sig)
        if cost <= budget:
            pkg.dependency_signatures[rel_path] = sig
            budget -= cost
        else:
            pkg.notes.append(f"{rel_path} signatures dropped (exceeds remaining budget)")
            pkg.truncated = True

    return pkg


def render_context_for_prompt(pkg: ContextPackage) -> str:
    parts = ["## Diff\n```diff\n" + pkg.diff_text + "\n```"]

    if pkg.changed_files:
        parts.append("## Full content of changed files")
        for path, content in pkg.changed_files.items():
            parts.append(f"### {path}\n```python\n{content}\n```")

    if pkg.dependency_signatures:
        parts.append("## Related local modules (signatures only, for interface context)")
        for path, sig in pkg.dependency_signatures.items():
            parts.append(f"### {path} (signatures)\n```python\n{sig}\n```")

    if pkg.truncated:
        parts.append(
            "## Context notes\n" + "\n".join(f"- {n}" for n in pkg.notes)
        )

    return "\n\n".join(parts)