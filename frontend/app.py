"""
frontend/app.py

Minimal Flask backend. The browser never sees a GitHub token — this server
holds it and calls the GitHub REST API to dispatch the workflow. This is
what solves your blocked-network problem: the actual LLM calls happen on
the GitHub Actions runner, not on your machine or the browser.

Run:
    export GITHUB_DISPATCH_TOKEN=ghp_xxx   # PAT with 'repo' + 'workflow' scope
    python frontend/app.py

Then open http://localhost:5000
"""

import os

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

REPO_FULL_NAME = os.environ.get("REPO_FULL_NAME", "testing-uses/code-review-agent-testing")
WORKFLOW_FILE = os.environ.get("WORKFLOW_FILE", "dev_agent_pipeline.yml")
GITHUB_API_BASE = "https://api.github.com"


def github_headers() -> dict:
    token = os.environ["GITHUB_DISPATCH_TOKEN"]
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


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
    url = f"{GITHUB_API_BASE}/repos/{REPO_FULL_NAME}/actions/workflows/{WORKFLOW_FILE}/runs"
    response = requests.get(url, headers=github_headers(), params={"per_page": 5})
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
