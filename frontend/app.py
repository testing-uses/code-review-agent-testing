import json
import os
import re
import zipfile
from io import BytesIO

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

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


def github_headers():
    token = os.environ.get("GITHUB_DISPATCH_TOKEN")

    if not token:
        raise RuntimeError(
            "GITHUB_DISPATCH_TOKEN is not configured. "
            "Set it in PowerShell before starting Flask."
        )

    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def github_get(path, params=None):
    response = requests.get(
        f"{GITHUB_API_BASE}{path}",
        headers=github_headers(),
        params=params,
        timeout=30,
    )
    return response


@app.errorhandler(RuntimeError)
def handle_runtime_error(error):
    return jsonify({"error": str(error)}), 500


@app.route("/")
def index():
    return render_template("index.html")


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
            "inputs": {
                "task_text": task_text,
            },
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

    runs_data = response.json().get("workflow_runs", [])

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
        for run in runs_data
    ])


def fetch_job_log_text(job_id):
    response = requests.get(
        f"{GITHUB_API_BASE}/repos/{REPO_FULL_NAME}"
        f"/actions/jobs/{job_id}/logs",
        headers=github_headers(),
        allow_redirects=True,
        timeout=60,
    )

    if response.status_code != 200:
        return (
            f"[frontend] Could not fetch job logs. "
            f"HTTP {response.status_code}: {response.text}"
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


def extract_log_lines(log_text):
    return [
        line
        for line in log_text.splitlines()
        if line.strip()
    ]


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
        events = parse_pipeline_events(log_text)

        all_events.extend(events)
        all_logs.append({
            "job_id": job["id"],
            "job_name": job["name"],
            "lines": extract_log_lines(log_text),
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