"""
agents/knowledge_base/build_kb.py

CLI entrypoint to build/update the central knowledge base. Run this once
in CI before any agent runs (or as a separate scheduled/merge-triggered
job), so every agent queries an up-to-date, shared KB instead of
re-parsing the repository independently.

Usage:
    python agents/knowledge_base/build_kb.py --repo-root . --db-path agents/knowledge_base/kb.sqlite3
"""

import argparse
import json

from kb_indexer import build_or_update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--db-path", default="agents/knowledge_base/kb.sqlite3")
    args = parser.parse_args()

    stats = build_or_update(args.repo_root, args.db_path)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
