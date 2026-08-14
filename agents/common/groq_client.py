"""
agents/common/groq_client.py

Shared Groq client used by every agent (Dev Agent, Code Review Agent,
future QA Agent). Centralizing this avoids each agent re-implementing its
own key-pool/retry/token-estimation logic with subtly different bugs.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict
from pathlib import Path
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


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN_ESTIMATE))


class GroqKeyPool:
    """Rotates across GROQ_API_KEY_1..N on rate limits / transient failures."""

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
            raise RuntimeError("No Groq API keys configured (GROQ_API_KEY_1..5).")

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


def preflight_check(system_prompt: str, user_prompt: str, max_output_tokens: int, ceiling: int) -> int:
    estimated = (
        estimate_tokens(system_prompt)
        + estimate_tokens(user_prompt)
        + max_output_tokens
    )
    if estimated > ceiling:
        raise ValueError(
            f"Prompt too large: estimated {estimated} tokens exceeds ceiling {ceiling}. "
            f"Reduce context or lower the DCBA-allocated budget before calling the model."
        )
    return estimated


def call_groq_json(
    key_pool: GroqKeyPool,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    token_ceiling: int = 8000,
) -> Dict[str, Any]:
    """Call Groq with JSON-mode output and automatic key fallback.
    Returns the parsed JSON plus the actual token usage for metrics."""
    preflight_check(system_prompt, user_prompt, max_output_tokens, token_ceiling)

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

            usage = getattr(response, "usage", None)
            result["_usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "key_index": key_index,
            }
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


def load_prompt(prompts_dir: str, filename: str) -> str:
    """Load a prompt from a standalone .md file — prompts must never be
    hardcoded strings inside agent logic."""
    path = os.path.join(prompts_dir, filename)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()