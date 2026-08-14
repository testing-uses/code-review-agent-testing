"""
agents/knowledge_base/schema.py

SQLite schema for the central knowledge base. This is the POC-scale stand-in
for a full knowledge-graph + vector-DB stack (e.g. Neo4j + a managed vector
store). At three files this is more than enough; the schema is designed so
migrating to Neo4j/pgvector later means writing new adapters, not
redesigning the data model.

Tables:
    files    -- one row per source file, keyed by git blob SHA for
                incremental re-indexing (skip unchanged files).
    symbols  -- functions/classes defined in each file, with a lightweight
                term-frequency "vector" (JSON) standing in for embeddings.
    edges    -- CALLS / IMPORTS relationships between symbols/files, used
                for reverse-dependency checks and PageRank.
"""

import sqlite3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    blob_sha TEXT NOT NULL,
    last_indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    file TEXT NOT NULL,
    kind TEXT NOT NULL,           -- 'function' | 'class'
    signature TEXT NOT NULL,
    docstring TEXT,
    term_vector TEXT,             -- JSON: {term: frequency}
    UNIQUE(name, file)
);

CREATE TABLE IF NOT EXISTS edges (
    src_symbol TEXT NOT NULL,     -- caller/importer symbol or file name
    dst_symbol TEXT NOT NULL,     -- callee/imported symbol or file name
    edge_type TEXT NOT NULL,      -- 'calls' | 'imports'
    src_file TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_symbol);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_symbol);
"""


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    return conn
