"""
review_agent/repo_map.py

A cached, incrementally-updated symbol index — same idea as Aider's repo map:
a compact index of function/class signatures across the repo, built once
and reused, rather than re-parsed in full on every PR.

Cache invalidation is per-file, keyed on the file's git blob SHA, so only
files that actually changed since the last cache write get re-parsed.

This is intentionally simple (regex/AST over local .py files) — swap in
tree-sitter or a proper import-graph resolver if the repo grows.
"""

import ast
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, List, Set

CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".review_agent_cache", "repo_map.json"
)


@dataclass
class Symbol:
    name: str
    file: str
    signature: str
    kind: str  # "function" | "class"


def _git_blob_sha(repo_root: str, rel_path: str) -> str:
    result = subprocess.run(
        ["git", "hash-object", rel_path],
        cwd=repo_root, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _extract_symbols(file_path: str, rel_path: str) -> List[Symbol]:
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return []

    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ", ".join(a.arg for a in node.args.args)
            symbols.append(Symbol(
                name=node.name,
                file=rel_path,
                signature=f"def {node.name}({args})",
                kind="function",
            ))
        elif isinstance(node, ast.ClassDef):
            symbols.append(Symbol(
                name=node.name,
                file=rel_path,
                signature=f"class {node.name}",
                kind="class",
            ))
    return symbols


class RepoMap:
    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        self._cache: Dict[str, dict] = {}  # rel_path -> {sha, symbols: [...]}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "r", encoding="utf-8") as fh:
                self._cache = json.load(fh)

    def _save_cache(self):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(self._cache, fh, indent=2)

    def refresh(self, exclude_dirs: Set[str] = frozenset({".git", "review_agent", ".review_agent_cache"})):
        """Rebuild only entries whose git blob SHA changed since last run."""
        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for filename in files:
                if not filename.endswith(".py"):
                    continue
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, self.repo_root)
                current_sha = _git_blob_sha(self.repo_root, rel_path)

                cached_entry = self._cache.get(rel_path)
                if cached_entry and cached_entry.get("sha") == current_sha:
                    continue  # unchanged — skip re-parsing

                symbols = _extract_symbols(full_path, rel_path)
                self._cache[rel_path] = {
                    "sha": current_sha,
                    "symbols": [asdict(s) for s in symbols],
                }

        self._save_cache()

    def resolve_symbols(self, identifiers: Set[str], exclude_files: Set[str]) -> List[Symbol]:
        """Return signatures for symbols whose name matches an identifier
        found in the diff, skipping files already covered elsewhere."""
        matches = []
        seen_names = set()

        for rel_path, entry in self._cache.items():
            if rel_path in exclude_files:
                continue
            for symbol_dict in entry.get("symbols", []):
                name = symbol_dict["name"]
                if name in identifiers and name not in seen_names:
                    matches.append(Symbol(**symbol_dict))
                    seen_names.add(name)

        return matches