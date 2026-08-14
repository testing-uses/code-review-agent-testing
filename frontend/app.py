"""
frontend/app.py

Minimal Flask backend. The browser never sees a GitHub token -- this server
holds it and calls the GitHub REST API to dispatch the workflow and to pull
back live progress. LLM calls happen on the GitHub Actions runner, not on
your machine or the browser.

New in this version:
- /status/<run_id> fetches the run, its jobs, and the job logs, then
  parses out [PIPELINE] JSON lines emitted by agents/orchestrator/progress.py
  so the frontend can render a real timeline instead of just a status blob.

Run:
    export GITHUB_DISPATCH_TOKEN=ghp_xxx   # PAT with 'repo' + 'workflow' scope
    python frontend/app.py

Then open http://localhost:5000
"""

import io
import json
import os
import re
import zipfile

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

REPO_FULL_NAME = os.environ.get("REPO_FULL_NAME", "testing-uses/code-review-agent-testing")
WORKFLOW_FILE = os.environ.get("WORKFLOW_FILE", "dev_agent_pipeline.yml")
GITHUB_API_BASE = "https://api.github.com"

PIPELINE_PREFIX = "[PIPELINE] "
PIPELINE_LINE_RE = re.compile(re.escape(PIPELINE_PREFIX) + r"(\{.*\})")


def github_headers() -> dict:
    token = os.environ["GITHUB_DISPATCH_TOKEN"]
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def github_get(path: str, params: dict | None = None) -> requests.Response:
    return requests.get(f"{GITHUB_API_BASE}{path}", headers=github_headers(), params=params)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/trigger", methods=["POST"])
def trigger():
    task_text = request.json.get("task_text", "").strip()
    if not task_text:
        return jsonify({"error": "task_text is required"}), 400

    url = f"{GITHUB_API_BASE}/repos/{REPO_FULL_NAME}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    response = requests.post(
        url, headers=github_headers(),
        json={"ref": "main", "inputs": {"task_text": task_text}},
    )

    if response.status_code == 204:
        return jsonify({"status": "dispatched"})
    return jsonify({"error": response.text}), response.status_code


@app.route("/runs")
def runs():
    response = github_get(
        f"/repos/{REPO_FULL_NAME}/actions/workflows/{WORKFLOW_FILE}/runs",
        params={"per_page": 5},
    )
    if response.status_code != 200:
        return jsonify({"error": response.text}), response.status_code

    runs_data = response.json().get("workflow_runs", [])
    simplified = [
        {
            "id": run["id"],
            "status": run["status"],
            "conclusion": run["conclusion"],
            "created_at": run["created_at"],
            "html_url": run["html_url"],
        }
        for run in runs_data
    ]
    return jsonify(simplified)


def _fetch_job_log_text(job_id: int) -> str:
    """GitHub returns a redirect to a plaintext (sometimes gzip) log blob."""
    url = f"{GITHUB_API_BASE}/repos/{REPO_FULL_NAME}/actions/jobs/{job_id}/logs"
    response = requests.get(url, headers=github_headers(), allow_redirects=True)
    if response.status_code != 200:
        return ""
    return response.text


def _parse_pipeline_events(log_text: str) -> list[dict]:
    events = []
    for line in log_text.splitlines():
        match = PIPELINE_LINE_RE.search(line)
        if not match:
            continue
        try:
            event = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        events.append(event)
    return events


def _dedupe_events(events: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for event in events:
        key = (event.get("event"), event.get("timestamp"), json.dumps(event, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    unique.sort(key=lambda e: e.get("timestamp", ""))
    return unique


@app.route("/status/<int:run_id>")
def status(run_id):
    run_response = github_get(f"/repos/{REPO_FULL_NAME}/actions/runs/{run_id}")
    if run_response.status_code != 200:
        return jsonify({"error": run_response.text}), run_response.status_code
    run = run_response.json()

    jobs_response = github_get(f"/repos/{REPO_FULL_NAME}/actions/runs/{run_id}/jobs")
    jobs_data = jobs_response.json().get("jobs", []) if jobs_response.status_code == 200 else []

    all_events = []
    job_summaries = []
    for job in jobs_data:
        log_text = _fetch_job_log_text(job["id"])
        all_events.extend(_parse_pipeline_events(log_text))
        job_summaries.append({
            "id": job["id"],
            "name": job["name"],
            "status": job["status"],
            "conclusion": job["conclusion"],
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
        })

    return jsonify({
        "run": {
            "id": run["id"],
            "status": run["status"],
            "conclusion": run["conclusion"],
            "html_url": run["html_url"],
            "created_at": run["created_at"],
        },
        "jobs": job_summaries,
        "events": _dedupe_events(all_events),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)