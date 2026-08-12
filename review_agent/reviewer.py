"""
review_agent/reviewer.py  (v2 — token-optimized)

Changes from v1:
  - System prompts shortened (no repeated schema prose across both passes;
    kept to what actually changes model behavior).
  - Rubric rendered as one compact line per category ("name(weight): focus"),
    not full-sentence descriptions, in the prompt itself.
  - Static-analysis output hard-capped in length.
  - Output token budgets lowered (review pass and verifier pass no longer
    request more tokens than the decision engine actually needs).
  - A pre-flight token estimate is computed before every API call; if it
    would exceed the safe ceiling, the caller degrades context further
    BEFORE sending (see run_review.py), instead of finding out via a 413.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any, Dict, List, Tuple

import yaml
from groq import APIError, APIStatusError, Groq, RateLimitError

RUBRIC_PATH = os.path.join(os.path.dirname(__file__), "rubric.yaml")

DEFAULT_MODEL = "llama-3.3-70b-versatile"

REVIEW_MAX_OUTPUT_TOKENS = 1200
VERIFIER_MAX_OUTPUT_TOKENS = 900

# Groq's org-level TPM limit counts prompt + completion together, so we cap
# the whole request (system + user + expected output) well under it.
SAFE_REQUEST_TOKEN_CEILING = 8000
CHARS_PER_TOKEN_ESTIMATE = 3.3

STATIC_ANALYSIS_MAX_CHARS = 800


class GroqKeyPool:
    def __init__(self) -> None:
        self.keys = [
            os.getenv("GROQ_API_KEY_1"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"),
        ]
        self.keys = [key for key in self.keys if key]

        if not self.keys:
            raise RuntimeError("No Groq API keys configured.")

        self.current_index = 0

    def clients_in_order(self):
        key_count = len(self.keys)
        for offset in range(key_count):
            index = (self.current_index + offset) % key_count
            yield index, Groq(api_key=self.keys[index])

    def mark_success(self, key_index: int) -> None:
        self.current_index = key_index

    def rotate_after_failure(self, key_index: int) -> None:
        self.current_index = (key_index + 1) % len(self.keys)


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN_ESTIMATE))


def load_rubric() -> Dict[str, Any]:
    with open(RUBRIC_PATH, "r", encoding="utf-8") as file:
        rubric = yaml.safe_load(file)
    if not rubric or "categories" not in rubric:
        raise ValueError("Invalid rubric.yaml: categories are missing.")
    return rubric


def render_rubric_compact(rubric: Dict[str, Any]) -> str:
    """One line per category instead of full prose descriptions."""
    lines = []
    for name, config in rubric["categories"].items():
        focus = config["description"].strip().split(".")[0]
        lines.append(f"- {name} (weight {config['weight']}): {focus}")
    return "\n".join(lines)


def run_static_analysis(repo_root: str, changed_files: List[str]) -> str:
    python_files = [f for f in changed_files if f.endswith(".py")]
    if not python_files:
        return "No Python files changed."

    reports: List[str] = []
    for command, tool_name in [
        (["ruff", "check", *python_files], "ruff"),
        (["bandit", "-q", *python_files], "bandit"),
    ]:
        try:
            result = subprocess.run(
                command, cwd=repo_root, capture_output=True,
                text=True, timeout=30, check=False,
            )
            output = "\n".join(
                part.strip() for part in [result.stdout, result.stderr] if part.strip()
            ) or "No issues found."
        except FileNotFoundError:
            output = f"{tool_name} not installed."
        except subprocess.TimeoutExpired:
            output = f"{tool_name} timed out."
        except Exception as error:
            output = f"{tool_name} failed: {error}"

        reports.append(f"### {tool_name}\n{output}")

    combined = "\n\n".join(reports)
    if len(combined) > STATIC_ANALYSIS_MAX_CHARS:
        combined = combined[:STATIC_ANALYSIS_MAX_CHARS] + "\n...(truncated)"
    return combined

REVIEWER_SYSTEM_PROMPT = """
You are a rigorous senior software engineer reviewing a pull request.

Review only the changed application code and the directly relevant context.
Ground every finding in code that is actually shown. Do not invent files,
requirements, behavior, or vulnerabilities.

Evaluate every rubric category provided by the user.

Report only actionable issues involving:

- Correctness
- Security
- Error handling
- Maintainability
- Documentation
- Performance
- Backward compatibility
- Architecture consistency

Do not report the absence of tests, test coverage, QA validation, or test
files. Testing is handled by a separate system.

Do not report:
- Personal style preferences.
- Issues unsupported by the supplied code.
- Duplicate findings.
- Hypothetical issues that are impossible under the current constants,
  control flow, or configuration.
- Generic recommendations without a concrete code reference.

Severity levels:

CRITICAL:
    Exploitable security issue, severe data corruption, or crash during
    normal use.

HIGH:
    Likely functional bug, serious security issue, or major compatibility risk.

MEDIUM:
    Real issue that should be addressed but is not immediately blocking.

LOW:
    Minor maintainability or documentation issue.

SCORING RULES (follow these exactly — do not deviate):

Each category's score depends ONLY on findings that belong to THAT
category. A finding in one category must never lower the score of a
different category. Categories are scored completely independently
of each other.

Use these fixed anchors for each category's score, based on the single
worst finding reported within that category:

- No findings in this category            -> score = 100
- Worst finding in this category is LOW    -> score between 90 and 99
- Worst finding in this category is MEDIUM -> score between 70 and 89
- Worst finding in this category is HIGH   -> score between 40 and 69
- Worst finding in this category is CRITICAL -> score between 0 and 39

Never assign 0 to a category unless that specific category contains a
CRITICAL finding. Never assign the same low score to every category just
because one category has a problem. A clean category with zero findings
must always score 100 — not 0, not "pending", not left unscored.

Every one of the 8 rubric categories must appear in your JSON output,
even if its findings list is empty and its score is 100.

WORKED EXAMPLE (follow this exact pattern of independent scoring):

If the diff has one HIGH bug in a function and one LOW trailing-whitespace
issue in an unrelated file, the correct output assigns:
  "correctness": {"score": 55, "findings": [the HIGH bug]}
  "maintainability_and_style": {"score": 95, "findings": [the LOW whitespace issue]}
  "security": {"score": 100, "findings": []}
  "error_handling": {"score": 100, "findings": []}
  "documentation": {"score": 100, "findings": []}
  "performance_and_resources": {"score": 100, "findings": []}
  "backward_compatibility": {"score": 100, "findings": []}
  "architecture_consistency": {"score": 100, "findings": []}

Notice that only "correctness" and "maintainability_and_style" are
affected. The other six categories remain at 100 because they have no
findings. This is the ONLY correct pattern. Do not lower unrelated
categories.

Return only valid JSON:

{
  "categories": {
    "<category_name>": {
      "score": 0,
      "findings": [
        {
          "severity": "CRITICAL | HIGH | MEDIUM | LOW",
          "file": "path/to/file.py",
          "line": 1,
          "title": "Short issue title",
          "explanation": "Evidence-based explanation",
          "recommendation": "Specific suggested fix"
        }
      ]
    }
  },
  "overall_summary": "Two or three sentence summary"
}

"""

VERIFIER_SYSTEM_PROMPT = """
You are a skeptical verifier for an automated code review.

Re-check every finding against the actual code context.

Classify each finding as:

confirmed:
    The issue is real and the original severity is appropriate.

downgraded:
    The issue is real but the severity should be lower.

discarded:
    The issue is unsupported, speculative, duplicated, or based on code
    that is not present.

Do not confirm hypothetical issues that are impossible under the current
constant values, control flow, or configuration.

Do not create findings about missing tests, test coverage, QA validation,
or test files. Testing is handled separately.

Return only valid JSON:

{
  "verified_findings": [
    {
      "severity": "CRITICAL | HIGH | MEDIUM | LOW",
      "file": "path/to/file.py",
      "line": 1,
      "title": "Short issue title",
      "explanation": "Evidence-based explanation",
      "recommendation": "Specific suggested fix",
      "verification_status": "confirmed | downgraded | discarded",
      "verification_note": "Why this status was selected"
    }
  ]
}
"""

def preflight_check(system_prompt: str, user_prompt: str, max_output_tokens: int) -> int:
    """Estimate total request tokens; raise before calling the API if this
    would exceed the safe ceiling, so callers can shrink context first."""
    estimated = (
        estimate_tokens(system_prompt)
        + estimate_tokens(user_prompt)
        + max_output_tokens
    )
    if estimated > SAFE_REQUEST_TOKEN_CEILING:
        raise ValueError(
            f"Prompt too large: estimated {estimated} tokens exceeds "
            f"safe ceiling {SAFE_REQUEST_TOKEN_CEILING}. Reduce context "
            f"before calling the model."
        )
    return estimated


def ensure_all_categories_present(
    rubric: Dict[str, Any],
    reviewer_output: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Guarantee every rubric category exists in the LLM output.
    If the model omitted a category entirely (commonly happens when it finds
    no issues in that category), backfill it with a clean 100/no-findings
    entry rather than letting it silently default to 0 downstream.
    """
    categories = reviewer_output.get("categories", {})
    if categories is None:
        categories = {}

    for category_name in rubric["categories"]:
        if category_name not in categories or not isinstance(categories[category_name], dict):
            categories[category_name] = {"score": 100, "findings": []}
        else:
            entry = categories[category_name]
            if "score" not in entry or not isinstance(entry.get("score"), (int, float)):
                entry["score"] = 100
            if "findings" not in entry or not isinstance(entry.get("findings"), list):
                entry["findings"] = []

    reviewer_output["categories"] = categories
    return reviewer_output

def run_reviewer_pass(
    key_pool: GroqKeyPool,
    model: str,
    rubric: Dict[str, Any],
    context_text: str,
    static_analysis_report: str,
) -> Dict[str, Any]:
    user_prompt = (
        f"## Rubric\n{render_rubric_compact(rubric)}\n\n"
        f"## Static analysis\n{static_analysis_report}\n\n"
        f"## Code context\n{context_text}"
    )
    reviewer_output = call_groq_json(
        key_pool, model, REVIEWER_SYSTEM_PROMPT, user_prompt, REVIEW_MAX_OUTPUT_TOKENS
    )
    return ensure_all_categories_present(rubric, reviewer_output)

def call_groq_json(
    key_pool: GroqKeyPool,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
) -> Dict[str, Any]:
    preflight_check(system_prompt, user_prompt, max_output_tokens)

    last_error: Exception | None = None

    for key_index, client in key_pool.clients_in_order():
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=max_output_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Groq returned an empty response.")
            result = json.loads(content)
            key_pool.mark_success(key_index)
            return result

        except RateLimitError as error:
            last_error = error
            key_pool.rotate_after_failure(key_index)
            time.sleep(1)
            continue

        except APIStatusError as error:
            last_error = error
            if error.status_code in {401, 413, 429, 500, 502, 503, 504}:
                key_pool.rotate_after_failure(key_index)
                time.sleep(1)
                continue
            raise

        except APIError as error:
            last_error = error
            key_pool.rotate_after_failure(key_index)
            time.sleep(1)
            continue

        except json.JSONDecodeError as error:
            raise RuntimeError("Groq returned invalid JSON.") from error

    raise RuntimeError(f"All Groq API keys failed. Last error: {last_error}")


def collect_findings(reviewer_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings = []
    for category_name, category_data in reviewer_output.get("categories", {}).items():
        for finding in category_data.get("findings", []):
            f = dict(finding)
            f["category"] = category_name
            findings.append(f)
    return findings


def run_verifier_pass(
    key_pool: GroqKeyPool,
    model: str,
    context_text: str,
    reviewer_output: Dict[str, Any],
) -> List[Dict[str, Any]]:
    findings = collect_findings(reviewer_output)
    if not findings:
        return []

    user_prompt = (
        f"## Code context\n{context_text}\n\n"
        f"## Findings to verify\n{json.dumps(findings)}"
    )

    result = call_groq_json(
        key_pool, model, VERIFIER_SYSTEM_PROMPT, user_prompt, VERIFIER_MAX_OUTPUT_TOKENS
    )

    verified = result.get("verified_findings", [])
    lookup: Dict[Tuple[str, str], str] = {
        (f.get("file", ""), f.get("title", "")): f.get("category", "unknown")
        for f in findings
    }

    final = []
    for v in verified:
        if v.get("verification_status") == "discarded":
            continue
        v["category"] = lookup.get((v.get("file", ""), v.get("title", "")), "unknown")
        final.append(v)
    return final


def review_pull_request(
    repo_root: str,
    changed_files: List[str],
    context_text: str,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    rubric = load_rubric()
    key_pool = GroqKeyPool()

    static_analysis_report = run_static_analysis(repo_root, changed_files)

    reviewer_output = run_reviewer_pass(
        key_pool, model, rubric, context_text, static_analysis_report
    )
    verified_findings = run_verifier_pass(key_pool, model, context_text, reviewer_output)

    category_scores = {
    name: reviewer_output.get("categories", {}).get(name, {}).get("score", 100)
    for name in rubric["categories"]
    }

    return {
        "rubric": rubric,
        "category_scores": category_scores,
        "verified_findings": verified_findings,
        "overall_summary": reviewer_output.get("overall_summary", ""),
        "static_analysis_report": static_analysis_report,
    }