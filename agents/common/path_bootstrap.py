"""
agents/common/path_bootstrap.py

Every agent file was independently doing its own version of:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "knowledge_base"))
    ...

...scattered across master_agent.py, dev_agent.py, context_selector.py,
run_review.py, reviewer.py, each computing its own relative path from its
own __file__. That's the direct cause of:

    ImportError: cannot import name 'select_context' from 'context_selector'

That specific message is the signature of a module ending up reachable
through two different sys.path entries that don't resolve to the exact
same string — Python then treats them as two separate module objects,
and whichever one gets imported second can be caught mid-initialization
(already in sys.modules, but not yet finished defining its functions) by
the first thing that reaches for a name on it.

Fix: one place computes AGENTS_ROOT as an absolute, resolved path and
adds every agent subdirectory to sys.path exactly once. Every entrypoint
imports and calls bootstrap() before any other agents.* import, instead
of hand-rolling sys.path.append calls.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_ROOT = os.path.dirname(_THIS_DIR)  # .../agents

_SUBDIRS = [
    "common",
    "dev_agent",
    "knowledge_base",
    "code_review_agent",
    "orchestrator",
]

_bootstrapped = False


def bootstrap() -> None:
    """Idempotent — safe to call from every entrypoint/module. Only
    mutates sys.path the first time it actually runs."""
    global _bootstrapped
    if _bootstrapped:
        return

    for subdir in _SUBDIRS:
        full_path = os.path.join(AGENTS_ROOT, subdir)
        if os.path.isdir(full_path) and full_path not in sys.path:
            sys.path.insert(0, full_path)

    _bootstrapped = True


bootstrap()