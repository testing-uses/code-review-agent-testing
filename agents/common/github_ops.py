"""
agents/common/github_ops.py

Git/GitHub operations owned by the ORCHESTRATOR, not the Dev Agent. This
matches the separation you described: the Dev Agent only produces file
content/patches; the orchestrator is what runs git and opens the PR.
"""

import os
import subprocess
from typing import List, Tuple

from github import Github


def commit_and_push(
    repo_root: str,
    branch_name: str,
    changed_files: List[str],
    commit_message: str,
) -> str:
    """Creates the branch, commits the given files, pushes it.
    Returns the new commit SHA (the PR's head SHA)."""
    subprocess.run(["git", "checkout", "-b", branch_name], cwd=repo_root, check=True)
    subprocess.run(["git", "add"] + changed_files, cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_root, check=True)
    subprocess.run(["git", "push", "-u", "origin", branch_name], cwd=repo_root, check=True)

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def create_pull_request(
    repo_full_name: str,
    branch_name: str,
    base_branch: str,
    title: str,
    body: str,
) -> Tuple[int, str]:
    """Returns (pr_number, head_sha)."""
    gh = Github(os.environ["GITHUB_TOKEN"])
    repo = gh.get_repo(repo_full_name)
    pr = repo.create_pull(title=title, body=body, head=branch_name, base=base_branch)
    return pr.number, pr.head.sha


def get_base_sha(repo_root: str, base_branch: str = "main") -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"origin/{base_branch}"],
        cwd=repo_root, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()
