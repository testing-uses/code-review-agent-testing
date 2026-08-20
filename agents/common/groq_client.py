"""agents/common/groq_client.py

Dedicated Google Gemini LLM Client.
Features:
- Multi-key rotation across GEMINI_API_KEY_1..3 and legacy GEMINI_API_KEY.
- Structured JSON output with Gemini OpenAPI schema sanitization.
- Automatic retry on rate limits (HTTP 429) and transient errors (HTTP 5xx).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE)

CHARS_PER_TOKEN_ESTIMATE = 3.3
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DEV_AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "required": [
        "blocked",
        "blocked_reason",
        "summary",
        "jira_key",
        "new_files",
        "diffs",
    ],
    "properties": {
        "blocked": {"type": "boolean"},
        "blocked_reason": {"type": "string"},
        "summary": {"type": "string"},
        "jira_key": {"type": "string"},
        "new_files": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
        "diffs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path", "diff"],
                "properties": {
                    "path": {"type": "string"},
                    "diff": {"type": "string"},
                },
            },
        },
    },
}


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN_ESTIMATE))


class GeminiKeyPool:
    """Rotate across GEMINI_API_KEY_1..3 and legacy GEMINI_API_KEY."""

    def __init__(self) -> None:
        raw_keys = [
            os.getenv("GEMINI_API_KEY_1"),
            os.getenv("GEMINI_API_KEY_2"),
            os.getenv("GEMINI_API_KEY_3"),
            os.getenv("GEMINI_API_KEY"),
        ]
        # Deduplicate while preserving order
        seen = set()
        self.keys: List[str] = []
        for k in raw_keys:
            if k and k.strip() and k.strip() not in seen:
                seen.add(k.strip())
                self.keys.append(k.strip())

        self.current_index = 0

    def has_keys(self) -> bool:
        return len(self.keys) > 0

    def keys_in_order(self) -> List[Tuple[int, str]]:
        key_count = len(self.keys)
        return [
            ((self.current_index + offset) % key_count, self.keys[(self.current_index + offset) % key_count])
            for offset in range(key_count)
        ]

    def mark_success(self, key_index: int) -> None:
        self.current_index = key_index

    def rotate_after_failure(self, key_index: int) -> None:
        if self.keys:
            self.current_index = (key_index + 1) % len(self.keys)


def preflight_check(
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    ceiling: int,
) -> int:
    estimated = (
        estimate_tokens(system_prompt)
        + estimate_tokens(user_prompt)
        + max_output_tokens
    )

    if estimated > ceiling:
        raise ValueError(
            f"Prompt too large: estimated {estimated} tokens exceeds "
            f"ceiling {ceiling}. Reduce context or lower the budget."
        )

    return estimated


def _convert_to_gemini_schema(schema: Optional[dict]) -> Optional[dict]:
    """Convert standard JSON Schema to Gemini's OpenAPI subset."""
    if not schema or not isinstance(schema, dict):
        return None

    gemini_schema: Dict[str, Any] = {}
    for k, v in schema.items():
        if k == "additionalProperties":
            continue  # Gemini REST API rejects additionalProperties
        if k == "type" and isinstance(v, str):
            gemini_schema["type"] = v.upper()
        elif k == "properties" and isinstance(v, dict):
            gemini_schema["properties"] = {
                pk: _convert_to_gemini_schema(pv) for pk, pv in v.items()
            }
        elif k == "items" and isinstance(v, dict):
            gemini_schema["items"] = _convert_to_gemini_schema(v)
        elif k == "required" and isinstance(v, list):
            gemini_schema["required"] = v
        elif k in ("description", "enum", "format", "nullable"):
            gemini_schema[k] = v
    return gemini_schema


def call_gemini_json(
    model: str = DEFAULT_GEMINI_MODEL,
    system_prompt: str = "",
    user_prompt: str = "",
    max_output_tokens: int = 2000,
    response_schema: Optional[dict] = None,
    key_pool: Optional[GeminiKeyPool] = None,
) -> Dict[str, Any]:
    """Call Google Gemini API with rotating GEMINI_API_KEY_1..3 keys."""
    if key_pool is None:
        key_pool = GeminiKeyPool()

    if not key_pool.has_keys():
        raise RuntimeError("No Gemini API keys configured (GEMINI_API_KEY_1..3 or GEMINI_API_KEY).")

    clean_model = model.replace("models/", "")
    last_error: Optional[Exception] = None

    for key_index, api_key in key_pool.keys_in_order():
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"

        payload: Dict[str, Any] = {
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]}
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.0,
                "maxOutputTokens": max_output_tokens,
            }
        }
        if system_prompt and system_prompt.strip():
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }
        gemini_schema = _convert_to_gemini_schema(response_schema)
        if gemini_schema:
            payload["generationConfig"]["responseSchema"] = gemini_schema

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if not candidates or "content" not in candidates[0]:
                    raise RuntimeError(f"Gemini API returned no candidate content: {data}")

                content_text = candidates[0]["content"]["parts"][0]["text"]
                result = json.loads(content_text)
                if not isinstance(result, dict):
                    raise RuntimeError("Gemini returned JSON, but it was not an object.")

                usage = data.get("usageMetadata", {})
                result["_usage"] = {
                    "prompt_tokens": usage.get("promptTokenCount"),
                    "completion_tokens": usage.get("candidatesTokenCount"),
                    "total_tokens": usage.get("totalTokenCount"),
                    "provider": "gemini",
                    "key_index": key_index,
                }
                key_pool.mark_success(key_index)
                return result

        except urllib.error.HTTPError as error:
            err_body = error.read().decode("utf-8")
            last_error = RuntimeError(f"Gemini HTTP {error.code} on key #{key_index + 1}: {err_body}")
            key_pool.rotate_after_failure(key_index)
            # Rotate and retry on 429 rate limit or 5xx server errors
            if error.code in (429, 500, 502, 503, 504):
                time.sleep(1)
                continue
            continue

        except Exception as error:
            last_error = error
            key_pool.rotate_after_failure(key_index)
            continue

    raise RuntimeError(f"All Gemini API keys failed after rotation. Last error: {last_error}")


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int = 2000,
    token_ceiling: int = 8000,
    response_schema: Optional[dict] = None,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    **kwargs,
) -> Dict[str, Any]:
    """Execute JSON generation using Google Gemini API."""
    preflight_check(
        system_prompt,
        user_prompt,
        max_output_tokens,
        token_ceiling,
    )
    return call_gemini_json(
        model=gemini_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_output_tokens=max_output_tokens,
        response_schema=response_schema,
    )


# Backward compatibility alias
def call_groq_json(*args, **kwargs):
    """Compatibility alias routing to call_gemini_json."""
    return call_llm_json(
        system_prompt=kwargs.get("system_prompt", ""),
        user_prompt=kwargs.get("user_prompt", ""),
        max_output_tokens=kwargs.get("max_output_tokens", 2000),
        token_ceiling=kwargs.get("token_ceiling", 8000),
        response_schema=kwargs.get("response_schema"),
    )


class GroqKeyPool(GeminiKeyPool):
    """Compatibility alias routing to GeminiKeyPool."""
    pass


def load_prompt(prompts_dir: str, filename: str) -> str:
    path = os.path.join(prompts_dir, filename)
    with open(path, "r", encoding="utf-8") as file_handle:
        return file_handle.read()