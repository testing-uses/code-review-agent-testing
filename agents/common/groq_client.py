"""agents/common/groq_client.py

Shared Groq client with model-key rotation, prompt-size checks, retries,
and optional strict JSON Schema output validation.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

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

DEV_AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "blocked",
        "blocked_reason",
        "summary",
        "jira_key",
        "diffs",
        "new_files",
    ],
    "properties": {
        "blocked": {"type": "boolean"},
        "blocked_reason": {"type": "string"},
        "summary": {"type": "string"},
        "jira_key": {"type": "string"},
        "diffs": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "new_files": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
}


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN_ESTIMATE))


class GroqKeyPool:
    """Rotate across GROQ_API_KEY_1..N on retryable failures."""

    def __init__(self) -> None:
        self.keys = [
            os.getenv("GROQ_API_KEY_1"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"),
            os.getenv("GROQ_API_KEY_4"),
            os.getenv("GROQ_API_KEY_5"),
        ]
        self.keys = [key for key in self.keys if key]

        if not self.keys:
            raise RuntimeError(
                "No Groq API keys configured (GROQ_API_KEY_1..5)."
            )

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
            "name": "dev_agent_edit",
            "strict": True,
            "schema": response_schema,
        },
    }


def call_groq_json(
    key_pool: GroqKeyPool,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    token_ceiling: int = 8000,
    response_schema: Optional[dict] = None,
) -> Dict[str, Any]:
    """Call Groq with optional strict JSON Schema output."""
    preflight_check(
        system_prompt,
        user_prompt,
        max_output_tokens,
        token_ceiling,
    )

    last_error: Optional[Exception] = None
    max_attempts = max(2, len(key_pool.keys) * 2)

    for attempt in range(max_attempts):
        clients = list(key_pool.clients_in_order())
        key_index, client = clients[attempt % len(clients)]

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
                response_format=_response_format(response_schema),
            )

            content = response.choices[0].message.content or ""
            if not content.strip():
                raise RuntimeError(
                    "Groq returned an empty JSON generation."
                )

            result = json.loads(content)
            if not isinstance(result, dict):
                raise RuntimeError(
                    "Groq returned JSON, but it was not an object."
                )

            key_pool.mark_success(key_index)
            usage = getattr(response, "usage", None)
            result["_usage"] = {
                "prompt_tokens": getattr(
                    usage,
                    "prompt_tokens",
                    None,
                ),
                "completion_tokens": getattr(
                    usage,
                    "completion_tokens",
                    None,
                ),
                "total_tokens": getattr(
                    usage,
                    "total_tokens",
                    None,
                ),
                "key_index": key_index,
                "attempt": attempt + 1,
            }
            return result

        except RateLimitError as error:
            last_error = error
            key_pool.rotate_after_failure(key_index)
            time.sleep(min(2 + attempt, 8))

        except APIStatusError as error:
            last_error = error
            if error.status_code in {
                400,
                401,
                413,
                429,
                500,
                502,
                503,
                504,
            }:
                key_pool.rotate_after_failure(key_index)
                time.sleep(min(2 + attempt, 8))
            else:
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


def load_prompt(prompts_dir: str, filename: str) -> str:
    path = os.path.join(prompts_dir, filename)
    with open(path, "r", encoding="utf-8") as file_handle:
        return file_handle.read()