"""
agents/orchestrator/metrics_store.py

Append-only JSONL metrics log. Every agent reports back to this after each
run, using a standardized schema, so the orchestrator's DCBA EMA correction
has real historical data to work with instead of guessing every time.
"""

import json
import os
from typing import Any, Dict, List, Optional


def record_metric(metrics_path: str, metric: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    with open(metrics_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(metric) + "\n")


def load_metrics(metrics_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(metrics_path):
        return []
    with open(metrics_path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def get_last_ema(metrics_path: str, agent_name: str) -> Optional[float]:
    """Return the most recent EMA-tracked token usage for this agent, if any."""
    entries = [
        m for m in load_metrics(metrics_path)
        if m.get("agent") == agent_name and "ema_total_tokens" in m
    ]
    if not entries:
        return None
    return entries[-1]["ema_total_tokens"]


def build_metric(
    agent: str,
    task_id: str,
    allocated_budget_tokens: int,
    actual_prompt_tokens: Optional[int],
    actual_completion_tokens: Optional[int],
    latency_ms: float,
    result_status: str,
    groq_key_used: Optional[int] = None,
    retry_count: int = 0,
    ema_total_tokens: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    total_tokens = (actual_prompt_tokens or 0) + (actual_completion_tokens or 0)
    metric = {
        "agent": agent,
        "task_id": task_id,
        "allocated_budget_tokens": allocated_budget_tokens,
        "actual_prompt_tokens": actual_prompt_tokens,
        "actual_completion_tokens": actual_completion_tokens,
        "total_tokens_used": total_tokens,
        "budget_utilization_pct": round(
            (total_tokens / allocated_budget_tokens) * 100, 1
        ) if allocated_budget_tokens else None,
        "latency_ms": latency_ms,
        "result_status": result_status,
        "groq_key_used": groq_key_used,
        "retry_count": retry_count,
        "ema_total_tokens": ema_total_tokens,
    }
    if extra:
        metric.update(extra)
    return metric
