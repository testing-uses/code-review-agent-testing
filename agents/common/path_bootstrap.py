"""
agents/common/path_bootstrap.py  (v2)

CHANGE from v1: v1 fixed the "same module reachable via two different
sys.path entries" ImportError, but exposed a second failure mode of the
same underlying problem — if a STALE DUPLICATE of a shared common module
(e.g. an old copy of patch_apply.py left in another agent directory)
exists anywhere under agents/, whichever directory sys.path happens to
search first wins, silently. That's what caused:

    NameError: name '_resolve_within_repo' is not defined

at `write_full_file()` — the traceback pointed at line 30 of the correct
NEW patch_apply.py, but the module actually bound to `patch_apply` in
sys.modules was a stale copy from elsewhere on the path that predates
the _resolve_within_repo helper being added. Nothing about the file
agents/common/patch_apply.py was wrong; a different file with the same
name shadowed it.

Two changes:
  1. `common/` is now inserted LAST (so it ends up FIRST in sys.path,
     i.e. highest priority) — genuinely shared modules living in
     common/ should always win a name collision against anything else.
  2. bootstrap() now actively scans every subdir for duplicate module
     names and prints a loud, actionable warning naming the exact
     duplicate files and which one will be used — so this fails loudly
     at startup instead of as a confusing NameError three calls deep.
"""

import os
import sys
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_ROOT = os.path.dirname(_THIS_DIR)  # .../agents

# Order matters: earlier entries here are inserted first, which — because
# bootstrap() does sys.path.insert(0, ...) for each — means they end up
# LAST in final sys.path priority. Put "common" last in this list so it
# ends up FIRST (highest priority) in sys.path.
_SUBDIRS = [
    "orchestrator",
    "code_review_agent",
    "knowledge_base",
    "dev_agent",
    "common",
]

_bootstrapped = False


def _scan_for_duplicate_modules() -> None:
    """Warn loudly if the same module filename exists under more than one
    agents/ subdirectory — that's exactly the shadowing bug that caused
    the write_full_file NameError. This doesn't fix a duplicate for you,
    it just makes sure you find out about it at startup instead of
    debugging a confusing runtime error later."""
    locations = defaultdict(list)
    for subdir in _SUBDIRS:
        full_dir = os.path.join(AGENTS_ROOT, subdir)
        if not os.path.isdir(full_dir):
            continue
        for filename in os.listdir(full_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                locations[filename].append(os.path.join(full_dir, filename))

    duplicates = {name: paths for name, paths in locations.items() if len(paths) > 1}
    if not duplicates:
        return

    print("[PIPELINE] " + __import__("json").dumps({
        "event": "duplicate_module_warning",
        "message": (
            "The same filename exists in more than one agents/ subdirectory. "
            "Whichever copy sys.path resolves first will silently shadow the "
            "others — this is the exact bug class behind stale-module "
            "NameErrors. Delete the stale copy, keep only the intended one."
        ),
        "duplicates": duplicates,
    }, sort_keys=True), flush=True)


def bootstrap() -> None:
    """Idempotent — safe to call from every entrypoint/module. Only
    mutates sys.path the first time it actually runs."""
    global _bootstrapped
    if _bootstrapped:
        return

    _scan_for_duplicate_modules()

    for subdir in _SUBDIRS:
        full_path = os.path.join(AGENTS_ROOT, subdir)
        if os.path.isdir(full_path) and full_path not in sys.path:
            sys.path.insert(0, full_path)

    _bootstrapped = True


bootstrap()