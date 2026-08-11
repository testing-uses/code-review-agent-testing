"""
review_agent/reviewer.py

Two-pass LLM code-review pipeline.

Pass 1:
    Reviews the pull request against the configured rubric.

Pass 2:
    Verifies every finding against the provided code context and removes
    unsupported or hallucinated findings.

The reviewer uses three Groq API keys with fallback handling.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any, Dict, List, Tuple

import yaml
from groq import APIError, APIStatusError, Groq, RateLimitError


RUBRIC_PATH = os.path.join(
    os.path.dirname(__file__),
    "rubric.yaml",
)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
REVIEW_MAX_OUTPUT_TOKENS = 3000
VERIFIER_MAX_OUTPUT_TOKENS = 2500


class GroqKeyPool:
    """
    Maintains a fallback pool of Groq clients.

    The pool tries the configured keys in order and rotates to the next key
    when a request receives a rate-limit or temporary API error.
    """

    def __init__(self) -> None:
        self.keys = [
            os.getenv("GROQ_API_KEY_1"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"),
        ]

        self.keys = [key for key in self.keys if key]

        if not self.keys:
            raise RuntimeError(
                "No Groq API keys found. Configure GROQ_API_KEY_1, "
                "GROQ_API_KEY_2, or GROQ_API_KEY_3."
            )

        self.current_index = 0

    def clients_in_order(self):
        """
        Yield Groq clients beginning at the current key.

        Example:
            Current key = 2
            Attempt order = key 2, key 3, key 1
        """
        key_count = len(self.keys)

        for offset in range(key_count):
            index = (self.current_index + offset) % key_count
            yield index, Groq(api_key=self.keys[index])

    def mark_success(self, key_index: int) -> None:
        """Use the successful key first for the next request."""
        self.current_index = key_index

    def rotate_after_failure(self, key_index: int) -> None:
        """Start the next request from the key after the failed key."""
        self.current_index = (key_index + 1) % len(self.keys)


def load_rubric() -> Dict[str, Any]:
    """Load the review rubric from rubric.yaml."""
    with open(RUBRIC_PATH, "r", encoding="utf-8") as file:
        rubric = yaml.safe_load(file)

    if not rubric or "categories" not in rubric:
        raise ValueError("Invalid rubric.yaml: categories are missing.")

    return rubric


def run_static_analysis(
    repo_root: str,
    changed_files: List[str],
) -> str:
    """
    Run deterministic static-analysis tools before calling the LLM.

    The LLM receives these results as evidence. Static analysis is not
    considered a QA/testing agent here.
    """
    python_files = [
        file_path
        for file_path in changed_files
        if file_path.endswith(".py")
    ]

    if not python_files:
        return "No Python files changed."

    reports: List[str] = []

    tools = [
        (["ruff", "check", *python_files], "ruff"),
        (["bandit", "-q", *python_files], "bandit"),
    ]

    for command, tool_name in tools:
        try:
            result = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            output = "\n".join(
                part.strip()
                for part in [result.stdout, result.stderr]
                if part.strip()
            )

            if not output:
                output = "No issues found."

        except FileNotFoundError:
            output = f"{tool_name} is not installed."

        except subprocess.TimeoutExpired:
            output = f"{tool_name} timed out after 30 seconds."

        except Exception as error:
            output = f"{tool_name} failed: {error}"

        reports.append(f"### {tool_name}\n{output}")

    return "\n\n".join(reports)


REVIEWER_SYSTEM_PROMPT = """
You are a rigorous senior software engineer reviewing a pull request.

Review only the code and context provided to you. Do not invent files,
functions, behavior, requirements, or vulnerabilities that are not supported
by the provided evidence.

Evaluate every rubric category.

For each category, return:
- A score from 0 to 100.
- A list of actionable findings.
- An empty findings list when no real issue exists.

Severity levels:

CRITICAL:
    Exploitable security issue, severe data corruption, or a crash during
    normal usage.

HIGH:
    Likely functional bug, serious security issue, or major compatibility risk.

MEDIUM:
    Real issue that should be addressed but is not immediately blocking.

LOW:
    Minor maintainability, style, or documentation issue.

Do not report:
- Personal stylistic preferences.
- Issues unsupported by the supplied code.
- Duplicate findings.
- Generic recommendations without a concrete code reference.

Return only valid JSON with this structure:

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
You are a skeptical code-review finding verifier.

You will receive:
1. The original code context.
2. Findings generated by another reviewer.

Re-check every finding against the actual code context.

For every finding, classify it as:

confirmed:
    The issue is real and the original severity is appropriate.

downgraded:
    The issue is real, but the severity should be lower. Return the corrected
    severity.

discarded:
    The issue is unsupported, incorrect, duplicated, or based on code that is
    not present in the context.

Be conservative. A finding must reference real code and explain a realistic
impact. Discard vague or speculative findings.

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


def call_groq_json(
    key_pool: GroqKeyPool,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
) -> Dict[str, Any]:
    """
    Call Groq and parse the response as JSON.

    Rate-limit and temporary API failures cause fallback to the next key.
    Invalid JSON is not retried because it indicates an output-format problem,
    not a key problem.
    """
    last_error: Exception | None = None

    for key_index, client in key_pool.clients_in_order():
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
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

            if error.status_code in {401, 429, 500, 502, 503, 504}:
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
            raise RuntimeError(
                "Groq returned invalid JSON. "
                "The response did not follow the required schema."
            ) from error

    raise RuntimeError(
        f"All Groq API keys failed. Last error: {last_error}"
    )


def run_reviewer_pass(
    key_pool: GroqKeyPool,
    model: str,
    rubric: Dict[str, Any],
    context_text: str,
    static_analysis_report: str,
) -> Dict[str, Any]:
    """Run the first-pass rubric-based review."""
    category_list = "\n".join(
        (
            f"- {name}: weight={config['weight']}, "
            f"focus={config['description'].strip()}"
        )
        for name, config in rubric["categories"].items()
    )

    user_prompt = f"""
## Review rubric

{category_list}

## Static-analysis output

{static_analysis_report}

## Code context

{context_text}

Evaluate the pull request strictly against the listed rubric categories.
Return every rubric category, even when its findings list is empty.
"""

    return call_groq_json(
        key_pool=key_pool,
        model=model,
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_output_tokens=REVIEW_MAX_OUTPUT_TOKENS,
    )


def collect_findings(
    reviewer_output: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Flatten category findings and attach their category names."""
    findings: List[Dict[str, Any]] = []

    for category_name, category_data in reviewer_output.get(
        "categories",
        {},
    ).items():
        for finding in category_data.get("findings", []):
            normalized_finding = dict(finding)
            normalized_finding["category"] = category_name
            findings.append(normalized_finding)

    return findings


def run_verifier_pass(
    key_pool: GroqKeyPool,
    model: str,
    context_text: str,
    reviewer_output: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Verify first-pass findings and remove discarded findings."""
    findings = collect_findings(reviewer_output)

    if not findings:
        return []

    user_prompt = f"""
## Original code context

{context_text}

## Findings to verify

{json.dumps(findings, indent=2)}
"""

    verifier_output = call_groq_json(
        key_pool=key_pool,
        model=model,
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_output_tokens=VERIFIER_MAX_OUTPUT_TOKENS,
    )

    verified_findings = verifier_output.get(
        "verified_findings",
        [],
    )

    category_lookup: Dict[Tuple[str, str], str] = {
        (
            finding.get("file", ""),
            finding.get("title", ""),
        ): finding.get("category", "unknown")
        for finding in findings
    }

    final_findings: List[Dict[str, Any]] = []

    for finding in verified_findings:
        status = finding.get("verification_status")

        if status == "discarded":
            continue

        lookup_key = (
            finding.get("file", ""),
            finding.get("title", ""),
        )

        finding["category"] = category_lookup.get(
            lookup_key,
            "unknown",
        )

        final_findings.append(finding)

    return final_findings


def review_pull_request(
    repo_root: str,
    changed_files: List[str],
    context_text: str,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Execute the complete two-pass review pipeline.

    The resulting dictionary is passed to decision_engine.py.
    """
    rubric = load_rubric()
    key_pool = GroqKeyPool()

    static_analysis_report = run_static_analysis(
        repo_root=repo_root,
        changed_files=changed_files,
    )

    reviewer_output = run_reviewer_pass(
        key_pool=key_pool,
        model=model,
        rubric=rubric,
        context_text=context_text,
        static_analysis_report=static_analysis_report,
    )

    verified_findings = run_verifier_pass(
        key_pool=key_pool,
        model=model,
        context_text=context_text,
        reviewer_output=reviewer_output,
    )

    category_scores = {
        category_name: category_data.get("score", 0)
        for category_name, category_data in reviewer_output.get(
            "categories",
            {},
        ).items()
    }

    return {
        "rubric": rubric,
        "category_scores": category_scores,
        "verified_findings": verified_findings,
        "overall_summary": reviewer_output.get(
            "overall_summary",
            "",
        ),
        "static_analysis_report": static_analysis_report,
    }