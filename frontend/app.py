"""
frontend/app.py

Flask backend for the Agentic AI Pipeline frontend. The browser never sees
a GitHub token -- this server holds it and calls the GitHub REST API to
dispatch workflows and pull back live progress.

CHANGE (balanced against the React rewrite in frontend/react/src/App.jsx):

1. Fixed the recurring route bug: /status/ was missing <int:run_id>,
   which meant fetch(`/status/${runId}`) from App.jsx would 404 every
   time. This is the same bug that broke the old Jinja-template frontend
   too -- fixing it here for good.

2. App.jsx is a separate Vite/React app, not a Jinja template. This file
   now supports BOTH ways of running it:

   a) DEV MODE (recommended while developing App.jsx): run Flask on
      :5000 and Vite's dev server separately on :5173. Flask-CORS is
      enabled so the Vite dev server can call this API cross-origin.
      Configure Vite's dev proxy (vite.config.js) to forward /trigger,
      /runs, /status/* to http://127.0.0.1:5000 if you want same-origin
      relative fetch() calls to just work without CORS at all -- either
      approach is fine, CORS is enabled here so it works even without
      the proxy.

   b) PRODUCTION / SINGLE-SERVER MODE: run `npm run build` in the React
      app, which outputs static files to frontend/react/dist. Flask then
      serves those files directly and App.jsx's relative fetch('/trigger'),
      fetch('/runs'), fetch('/status/...') calls hit this same Flask
      process with no CORS needed at all, because everything is
      same-origin.

   The JSON shapes returned by /runs and /status/<run_id> are UNCHANGED
   from the previous version -- App.jsx was written against this exact
   contract (run.html_url, jobs[].id/name/status/conclusion,
   events[], logs[].job_id/job_name/lines), so no field renaming needed.

Run (dev mode, with Vite running separately on :5173):
    export GITHUB_DISPATCH_TOKEN=ghp_xxx
    python frontend/app.py
    # in another terminal: cd frontend/react && npm run dev

Run (production mode, single server):
    cd frontend/react && npm run build
    export GITHUB_DISPATCH_TOKEN=ghp_xxx
    python frontend/app.py
    # open http://127.0.0.1:5000 -- Flask serves the built React app directly
"""

import json
import os
import zipfile
from io import BytesIO

import requests
from flask import Flask, jsonify, request, send_from_directory

try:
    from flask_cors import CORS
    _CORS_AVAILABLE = True
except ImportError:
    _CORS_AVAILABLE = False

REPO_FULL_NAME = os.environ.get(
    "REPO_FULL_NAME",
    "testing-uses/code-review-agent-testing",
)
WORKFLOW_FILE = os.environ.get(
    "WORKFLOW_FILE",
    "dev_agent_pipeline.yml",
)
GITHUB_API_BASE = "https://api.github.com"
PIPELINE_PREFIX = "[PIPELINE] "

# Where the built React app lives after `npm run build` inside
# frontend/react. If this directory doesn't exist yet (you haven't built
# the React app, or you're only running Vite's dev server separately),
# Flask simply won't have anything to serve at "/" -- that's fine in dev
# mode, since Vite serves the UI on its own port instead.
REACT_BUILD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "react",
    "dist",
)

app = Flask(__name__, static_folder=None)

# Only needed for DEV MODE (Vite on a different port calling this API
# cross-origin). Harmless to leave enabled in production mode too, since
# same-origin requests aren't affected by CORS headers either way.
if _CORS_AVAILABLE:
    CORS(
        app,
        resources={
            r"/trigger": {"origins": "*"},
            r"/runs": {"origins": "*"},
            r"/status/*": {"origins": "*"},
        },
    )


def github_headers():
    token = os.environ.get("GITHUB_DISPATCH_TOKEN")

    if not token:
        raise RuntimeError(
            "GITHUB_DISPATCH_TOKEN is not configured. "
            "Set it in the same terminal before starting Flask."
        )

    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def github_get(path, params=None):
    return requests.get(
        f"{GITHUB_API_BASE}{path}",
        headers=github_headers(),
        params=params,
        timeout=30,
    )


@app.errorhandler(RuntimeError)
def handle_runtime_error(error):
    return jsonify({"error": str(error)}), 500


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    app.logger.exception("Unhandled Flask error")
    return jsonify({
        "error": str(error),
        "type": type(error).__name__,
    }), 500


# ---- Serve the built React app (production mode) ----
# In dev mode (React app running via `npm run dev` on its own port), this
# route simply won't be hit for the UI -- you'd open Vite's dev server
# URL instead, and only the API routes below matter.
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react_app(path):
    if not os.path.isdir(REACT_BUILD_DIR):
        return jsonify({
            "error": (
                "React build not found at frontend/react/dist. "
                "Run `npm run build` inside frontend/react for production "
                "mode, or run the Vite dev server separately (npm run dev) "
                "and use its URL instead of this Flask server's root."
            ),
        }), 404

    requested_path = os.path.join(REACT_BUILD_DIR, path)
    if path and os.path.isfile(requested_path):
        return send_from_directory(REACT_BUILD_DIR, path)

    # SPA fallback -- any non-file path (client-side route) gets index.html
    return send_from_directory(REACT_BUILD_DIR, "index.html")


# ---- API routes used by App.jsx ----

@app.route("/trigger", methods=["POST"])
def trigger():
    body = request.get_json(silent=True) or {}
    task_text = body.get("task_text", "").strip()

    if not task_text:
        return jsonify({"error": "task_text is required"}), 400

    response = requests.post(
        (
            f"{GITHUB_API_BASE}/repos/{REPO_FULL_NAME}"
            f"/actions/workflows/{WORKFLOW_FILE}/dispatches"
        ),
        headers=github_headers(),
        json={
            "ref": "main",
            "inputs": {"task_text": task_text},
        },
        timeout=30,
    )

    if response.status_code != 204:
        return jsonify({
            "error": response.text,
            "status_code": response.status_code,
        }), response.status_code

    return jsonify({"status": "dispatched"})


@app.route("/runs")
def runs():
    response = github_get(
        f"/repos/{REPO_FULL_NAME}/actions/workflows/{WORKFLOW_FILE}/runs",
        params={"per_page": 10},
    )

    if response.status_code != 200:
        return jsonify({"error": response.text}), response.status_code

    workflow_runs = response.json().get("workflow_runs", [])

    return jsonify([
        {
            "id": run["id"],
            "status": run["status"],
            "conclusion": run["conclusion"],
            "created_at": run["created_at"],
            "updated_at": run.get("updated_at"),
            "html_url": run["html_url"],
            "head_branch": run.get("head_branch"),
        }
        for run in workflow_runs
    ])


def fetch_job_log_text(job_id):
    response = requests.get(
        (
            f"{GITHUB_API_BASE}/repos/{REPO_FULL_NAME}"
            f"/actions/jobs/{job_id}/logs"
        ),
        headers=github_headers(),
        allow_redirects=True,
        timeout=60,
    )

    if response.status_code != 200:
        return (
            f"[frontend] Could not fetch job logs: "
            f"HTTP {response.status_code} {response.text}"
        )

    content_type = response.headers.get("Content-Type", "").lower()

    if "zip" in content_type or response.content[:2] == b"PK":
        try:
            with zipfile.ZipFile(BytesIO(response.content)) as archive:
                parts = []

                for name in archive.namelist():
                    with archive.open(name) as log_file:
                        parts.append(
                            log_file.read().decode(
                                "utf-8",
                                errors="replace",
                            )
                        )

                return "\n".join(parts)

        except zipfile.BadZipFile:
            return response.text

    return response.text


def parse_pipeline_events(log_text):
    events = []

    for line in log_text.splitlines():
        marker_index = line.find(PIPELINE_PREFIX)

        if marker_index == -1:
            continue

        raw_payload = line[
            marker_index + len(PIPELINE_PREFIX):
        ].strip()

        try:
            event = json.loads(raw_payload)
        except json.JSONDecodeError:
            continue

        if isinstance(event, dict):
            events.append(event)

    return events


def deduplicate_events(events):
    seen = set()
    unique = []

    for event in events:
        key = json.dumps(
            event,
            sort_keys=True,
            default=str,
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(event)

    unique.sort(
        key=lambda event: event.get("timestamp", "")
    )

    return unique


# FIX: this was previously "/status/" with no <int:run_id> -- App.jsx's
# fetch(`/status/${runId}`) would 404 against that route every time.
@app.route("/status/<int:run_id>")
def status(run_id):
    run_response = github_get(
        f"/repos/{REPO_FULL_NAME}/actions/runs/{run_id}"
    )

    if run_response.status_code != 200:
        return jsonify({
            "error": run_response.text,
        }), run_response.status_code

    run = run_response.json()

    jobs_response = github_get(
        f"/repos/{REPO_FULL_NAME}/actions/runs/{run_id}/jobs",
        params={"per_page": 100},
    )

    if jobs_response.status_code != 200:
        return jsonify({
            "error": jobs_response.text,
        }), jobs_response.status_code

    jobs = jobs_response.json().get("jobs", [])
    all_events = []
    all_logs = []
    job_summaries = []

    for job in jobs:
        log_text = fetch_job_log_text(job["id"])

        all_events.extend(
            parse_pipeline_events(log_text)
        )

        all_logs.append({
            "job_id": job["id"],
            "job_name": job["name"],
            "lines": [
                line
                for line in log_text.splitlines()
                if line.strip()
            ],
        })

        job_summaries.append({
            "id": job["id"],
            "name": job["name"],
            "status": job["status"],
            "conclusion": job["conclusion"],
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "html_url": job.get("html_url"),
        })

    return jsonify({
        "run": {
            "id": run["id"],
            "status": run["status"],
            "conclusion": run["conclusion"],
            "created_at": run["created_at"],
            "updated_at": run.get("updated_at"),
            "html_url": run["html_url"],
        },
        "jobs": job_summaries,
        "events": deduplicate_events(all_events),
        "logs": all_logs,
    })


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000,
    )
