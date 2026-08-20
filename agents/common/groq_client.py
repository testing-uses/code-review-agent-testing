"""agents/common/groq_client.py

Unified LLM Provider client.
Primary: Google Gemini (supports 2-3 rotating API keys: GEMINI_API_KEY_1..3).
Fallback: Groq (supports rotating API keys: GROQ_API_KEY_1..5).
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
from groq import (
    APIConnectionError,
    APIError,
    APIStatusError,
    Groq,
    RateLimitError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE)

CHARS_PER_TOKEN_ESTIMATE = 3.3
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
GROQ_FALLBACK_MODELS = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
]


def _convert_to_gemini_schema(schema: Optional[dict]) -> Optional[dict]:
    """Convert standard JSON Schema to Gemini's OpenAPI subset."""
    if not schema or not isinstance(schema, dict):
        return None

    gemini_schema: Dict[str, Any] = {}
    for k, v in schema.items():
        if k == "additionalProperties":
            continue  # Gemini REST API does not support additionalProperties
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

DEV_AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
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
                "additionalProperties": False,
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
                "additionalProperties": False,
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


class GroqKeyPool:
    """Rotate across GROQ_API_KEY_1..5 on retryable failures."""

    def __init__(self) -> None:
        raw_keys = [
            os.getenv("GROQ_API_KEY_1"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"),
            os.getenv("GROQ_API_KEY_4"),
            os.getenv("GROQ_API_KEY_5"),
            os.getenv("GROQ_API_KEY"),
        ]
        seen = set()
        self.keys: List[str] = []
        for k in raw_keys:
            if k and k.strip() and k.strip() not in seen:
                seen.add(k.strip())
                self.keys.append(k.strip())

        self.current_index = 0

    def has_keys(self) -> bool:
        return len(self.keys) > 0

    def clients_in_order(self):
        key_count = len(self.keys)
        for offset in range(key_count):
            index = (self.current_index + offset) % key_count
            yield index, Groq(api_key=self.keys[index])

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


def _response_format(response_schema: Optional[dict]) -> dict:
    if response_schema is None:
        return {"type": "json_object"}

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_output",
            "strict": True,
            "schema": response_schema,
        },
    }


def _extract_result(response: Any, provider: str) -> Dict[str, Any]:
    content = response.choices[0].message.content or ""

    if not content.strip():
        raise RuntimeError(
            f"{provider} returned an empty JSON generation."
        )

    result = json.loads(content)
    if not isinstance(result, dict):
        raise RuntimeError(
            f"{provider} returned JSON, but it was not an object."
        )

    usage = getattr(response, "usage", None)
    result["_usage"] = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "provider": provider,
    }
    return result


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
            with urllib.request.urlopen(req, timeout=30) as resp:
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
            # Retry next key on 429 quota/rate limit or 5xx server errors
            if error.code in (429, 500, 502, 503, 504):
                time.sleep(1)
                continue
            # For bad request or invalid key, also rotate to try next available key
            continue

        except Exception as error:
            last_error = error
            key_pool.rotate_after_failure(key_index)
            continue

    raise RuntimeError(f"All Gemini API keys failed. Last error: {last_error}")


def call_groq_json(
    key_pool: Optional[GroqKeyPool] = None,
    model: str = DEFAULT_GROQ_MODEL,
    system_prompt: str = "",
    user_prompt: str = "",
    max_output_tokens: int = 2000,
    token_ceiling: int = 8000,
    response_schema: Optional[dict] = None,
) -> Dict[str, Any]:
    """Call Groq with rotating GROQ_API_KEY_1..5 keys and model fallbacks."""
    if key_pool is None:
        key_pool = GroqKeyPool()

    if not key_pool.has_keys():
        raise RuntimeError("No Groq API keys configured (GROQ_API_KEY_1..5).")

    preflight_check(
        system_prompt,
        user_prompt,
        max_output_tokens,
        token_ceiling,
    )

    last_error: Optional[Exception] = None
    candidate_models = [model] + [m for m in GROQ_FALLBACK_MODELS if m != model]
    max_attempts = max(2, len(key_pool.keys) * len(candidate_models))

    for attempt in range(max_attempts):
        clients = list(key_pool.clients_in_order())
        key_index, client = clients[attempt % len(clients)]
        current_model = candidate_models[(attempt // len(clients)) % len(candidate_models)]

        try:
            response = client.chat.completions.create(
                model=current_model,
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
                response_format=_response_format(response_schema),
            )

            result = _extract_result(response, "groq")
            result["_usage"]["key_index"] = key_index
            result["_usage"]["attempt"] = attempt + 1
            result["_usage"]["model"] = current_model
            key_pool.mark_success(key_index)
            return result

        except RateLimitError as error:
            last_error = error
            key_pool.rotate_after_failure(key_index)
            time.sleep(min(2 + attempt, 8))

        except APIStatusError as error:
            last_error = error

            if error.status_code == 404:
                # Model not found on this Groq account/tier -> try next model in candidate list
                continue

            if error.status_code == 413:
                raise RuntimeError(
                    "Groq rejected the request as too large. "
                    "Reduce prompt/output tokens; rotating API keys will "
                    "not reduce one request's size."
                ) from error

            if error.status_code in {400, 401}:
                raise RuntimeError(
                    f"Groq rejected the request (HTTP {error.status_code}): "
                    f"{error}"
                ) from error

            if error.status_code in {429, 500, 502, 503, 504}:
                key_pool.rotate_after_failure(key_index)
                time.sleep(min(2 + attempt, 8))
                continue

            raise

        except APIConnectionError as error:
            last_error = error
            key_pool.rotate_after_failure(key_index)
            time.sleep(min(2 + attempt, 8))

        except APIError as error:
            last_error = error
            key_pool.rotate_after_failure(key_index)
            time.sleep(min(2 + attempt, 8))

        except (RuntimeError, json.JSONDecodeError) as error:
            last_error = error
            key_pool.rotate_after_failure(key_index)
            time.sleep(min(2 + attempt, 8))

    raise RuntimeError(
        f"All Groq attempts failed after {max_attempts} attempts. "
        f"Last error: {last_error}"
    )


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int = 2000,
    token_ceiling: int = 8000,
    response_schema: Optional[dict] = None,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    groq_model: str = DEFAULT_GROQ_MODEL,
) -> Dict[str, Any]:
    """Primary: Gemini with key rotation. Fallback: Groq with key rotation."""
    gemini_pool = GeminiKeyPool()
    groq_pool = GroqKeyPool()

    # If Gemini keys are available, try Gemini first
    if gemini_pool.has_keys():
        try:
            return call_gemini_json(
                model=gemini_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=max_output_tokens,
                response_schema=response_schema,
                key_pool=gemini_pool,
            )
        except Exception as gemini_err:
            print(f"[LLM_FALLBACK] Gemini failed ({gemini_err}). Falling back to Groq...", flush=True)

    # Fallback to Groq
    if groq_pool.has_keys():
        return call_groq_json(
            key_pool=groq_pool,
            model=groq_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            token_ceiling=token_ceiling,
            response_schema=response_schema,
        )

    raise RuntimeError("No LLM provider keys configured. Set GEMINI_API_KEY_1..3 or GROQ_API_KEY_1..5 in .env")


def load_prompt(prompts_dir: str, filename: str) -> str:
    path = os.path.join(prompts_dir, filename)
    with open(path, "r", encoding="utf-8") as file_handle:
        return file_handle.read()