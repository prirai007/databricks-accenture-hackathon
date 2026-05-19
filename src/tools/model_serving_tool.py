"""LLM inference — Databricks Model Serving with OpenRouter fallback.

Primary: Databricks Model Serving (Qwen 3 80B via workspace endpoint).
Fallback: OpenRouter API (minimax-m2.1) when Databricks is unavailable.

Ref: https://docs.databricks.com/en/machine-learning/model-serving
Ref: https://openrouter.ai/docs
"""

import logging

import mlflow
import requests
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

from src.config import LLM_ENDPOINT, OPENROUTER_API_KEY, db_client

logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_MODEL = "minimax/minimax-m2.1"


def _call_openrouter(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> str:
    """Call OpenRouter API as LLM fallback."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "Databricks LLM failed and no OPENROUTER_API_KEY configured. "
            "Set OPENROUTER_API_KEY in .env to enable the fallback."
        )
    resp = requests.post(
        _OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": _OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return content if content is not None else ""


@mlflow.trace(name="query_llm", span_type="LLM")
def query_llm(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> str:
    """Call LLM for inference. Tries OpenRouter first, falls back to Databricks.

    Args:
        system_prompt: System-level instructions for the LLM.
        user_message: User-facing content to process.
        max_tokens: Maximum response length.
        temperature: Sampling temperature (lower = more deterministic).

    Returns:
        The LLM's text response.
    """
    # ── Primary: OpenRouter ──
    if OPENROUTER_API_KEY:
        try:
            return _call_openrouter(system_prompt, user_message, max_tokens, temperature)
        except Exception as e:
            logger.warning("OpenRouter failed: %s — trying Databricks", e)

    # ── Fallback: Databricks Model Serving ──
    if LLM_ENDPOINT:
        try:
            response = db_client.serving_endpoints.query(
                name=LLM_ENDPOINT,
                messages=[
                    ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
                    ChatMessage(role=ChatMessageRole.USER, content=user_message),
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("Databricks Model Serving failed: %s", e)

    raise RuntimeError("No LLM available. Set OPENROUTER_API_KEY in .env")
