"""
agents/knowledge_base/indexer.py

Incremental indexer: builds/updates the knowledge base by parsing changed
files only (git-blob-hash cache, same principle as repo_map.py in the
Code Review Agent, extended into a proper graph + term-vector store).

Term vectors here are simple term-frequency dicts, not real embeddings —
this is intentional for the POC: it gives "vector-DB-like" semantic search
without requiring an external embeddings API or extra dependencies. Swap
in real embeddings (e.g. a hosted embedding model) later without changing
the schema or query interface.
"""

import ast
import datetime
import json
import os
import re
import subprocess
from collections import Counter
from typing import Dict, List, Set, Tuple

from kb_schema import get_connection

STOPWORDS = {
    "self", "cls", "def", "return", "the", "and", "for", "with", "from", "import",
    "as", "if", "else", "elif", "none", "true", "false", "int", "str", "list",
    "dict", "set", "tuple", "bool", "float", "optional", "any", "union", "args",
    "kwargs", "class", "pass", "raise", "except", "try", "finally", "while",
    "break", "continue", "in", "not", "is", "or", "lambda", "yield", "async",
    "await", "init", "main", "type", "object",
}


def git_blob_sha(repo_root: str, rel_path: str) -> str:
    result = subprocess.run(
        ["git", "hash-object", rel_path],
        cwd=repo_root, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def tokenize(text: str) -> List[str]:
    raw_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
    tokens = []
    for tok in raw_tokens:
        tok_lower = tok.lower()
        if len(tok_lower) > 2 and tok_lower not in STOPWORDS:
            tokens.append(tok_lower)
        # Split camelCase and snake_case sub-tokens
        sub_tokens = re.findall(r"[a-z]+|[A-Z][a-z]*|\d+", tok)
        for st in sub_tokens:
            st_lower = st.lower()
            if len(st_lower) > 2 and st_lower != tok_lower and st_lower not in STOPWORDS:
                tokens.append(st_lower)
    return tokens


def term_vector(*text_parts: str) -> Dict[str, int]:
    tokens = []
    for part in text_parts:
        if part:
            tokens.extend(tokenize(part))
    return dict(Counter(tokens))


def extract_symbols_and_edges(
    source: str, rel_path: str,
) -> Tuple[List[dict], List[Tuple[str, str, str]]]:
    """Returns (symbols, edges) for one file.
    edges: (src_symbol_name, dst_symbol_name, edge_type)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []

    symbols: List[dict] = []
    edges: List[Tuple[str, str, str]] = []

    # imports -> edges from the file itself to the imported module
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append((rel_path, alias.name, "imports"))
        elif isinstance(node, ast.ImportFrom) and node.module:
            edges.append((rel_path, node.module, "imports"))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ", ".join(a.arg for a in node.args.args)
            docstring = ast.get_docstring(node) or ""
            symbols.append({
                "name": node.name,
                "kind": "function",
                "signature": f"def {node.name}({args})",
                "docstring": docstring,
            })
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                    edges.append((node.name, inner.func.id, "calls"))
                elif isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                    edges.append((node.name, inner.func.attr, "calls"))

        elif isinstance(node, ast.ClassDef):
            docstring = ast.get_docstring(node) or ""
            symbols.append({
                "name": node.name,
                "kind": "class",
                "signature": f"class {node.name}",
                "docstring": docstring,
            })

    return symbols, edges


def build_or_update(repo_root: str, db_path: str, exclude_dirs: Set[str] = None) -> dict:
    exclude_dirs = exclude_dirs or {".git", "agents", ".review_agent_cache", "__pycache__"}
    conn = get_connection(db_path)
    cursor = conn.cursor()

    stats = {"files_scanned": 0, "files_updated": 0, "files_skipped": 0, "symbols_indexed": 0}

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for filename in files:
            if not filename.endswith(".py"):
                continue

            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, repo_root)
            stats["files_scanned"] += 1

            current_sha = git_blob_sha(repo_root, rel_path)
            cursor.execute("SELECT blob_sha FROM files WHERE path = ?", (rel_path,))
            row = cursor.fetchone()

            if row and row[0] == current_sha:
                stats["files_skipped"] += 1
                continue

            with open(full_path, "r", encoding="utf-8") as fh:
                source = fh.read()

            symbols, edges = extract_symbols_and_edges(source, rel_path)

            cursor.execute("DELETE FROM symbols WHERE file = ?", (rel_path,))
            cursor.execute("DELETE FROM edges WHERE src_file = ?", (rel_path,))

            for symbol in symbols:
                vector = term_vector(symbol["name"], symbol["signature"], symbol["docstring"])
                cursor.execute(
                    "INSERT OR REPLACE INTO symbols (name, file, kind, signature, docstring, term_vector) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (symbol["name"], rel_path, symbol["kind"], symbol["signature"],
                     symbol["docstring"], json.dumps(vector)),
                )
                stats["symbols_indexed"] += 1

            for src, dst, edge_type in edges:
                cursor.execute(
                    "INSERT INTO edges (src_symbol, dst_symbol, edge_type, src_file) VALUES (?, ?, ?, ?)",
                    (src, dst, edge_type, rel_path),
                )

            cursor.execute(
                "INSERT OR REPLACE INTO files (path, blob_sha, last_indexed_at) VALUES (?, ?, ?)",
                (rel_path, current_sha, datetime.datetime.utcnow().isoformat()),
            )
            stats["files_updated"] += 1

    conn.commit()
    conn.close()
    return stats
