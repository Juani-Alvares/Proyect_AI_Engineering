"""Optional final synthesis and transparent token-cost estimation."""

import asyncio
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from src.config import (
    API_LLM_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    OPENAI_API_KEY,
)

LOGGER = logging.getLogger(__name__)
GEMINI_MAX_ATTEMPTS = 3
TRANSIENT_GEMINI_STATUS_CODES = {429, 500, 503}


def usage_and_cost(response: Any, model: str) -> dict[str, int | float | str]:
    """Estimates gpt-4o-mini cost; Phoenix remains the official cost evidence."""
    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    # Public rates expressed in USD per one million tokens.
    rates = (0.15, 0.60) if model == "gpt-4o-mini" else (0.0, 0.0)
    cost = input_tokens / 1_000_000 * rates[0] + output_tokens / 1_000_000 * rates[1]
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(cost, 8),
    }


def _synthesis_prompt(query: str, answer: str) -> str:
    return (
        "Resume con claridad el siguiente resultado técnico sin inventar datos. "
        f"Consulta: {query}\nResultado: {answer}"
    )


async def _openai_synthesis(prompt: str) -> tuple[str, dict]:
    model_name = LLM_MODEL or "gpt-4o-mini"
    model = ChatOpenAI(model=model_name, temperature=0, max_tokens=min(200, LLM_MAX_TOKENS))
    response = await model.ainvoke(prompt)
    return str(response.content), usage_and_cost(response, model_name)


def _gemini_usage(response: Any, model: str) -> dict[str, int | str]:
    """Returns usage reported by Gemini itself; it never invents a local cost."""
    usage = getattr(response, "usage_metadata", None)
    return {
        "model": model,
        "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
        "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
    }


def _gemini_config(types: Any) -> Any:
    """Uses low Gemini thinking while reserving enough output for a short answer."""
    return types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=max(512, min(1024, LLM_MAX_TOKENS)),
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.LOW,
        ),
    )


def _is_transient_gemini_error(error: Exception) -> bool:
    """Retries only documented temporary Gemini provider status codes."""
    try:
        return int(getattr(error, "code", 0)) in TRANSIENT_GEMINI_STATUS_CODES
    except (TypeError, ValueError):
        return False


async def _generate_gemini_with_retry(generate_content: Any, **request: Any) -> Any:
    """Retries 429/500/503 without blocking the event loop."""
    for attempt in range(GEMINI_MAX_ATTEMPTS):
        try:
            return await generate_content(**request)
        except Exception as error:
            if not _is_transient_gemini_error(error) or attempt == GEMINI_MAX_ATTEMPTS - 1:
                raise

            delay = 2 ** (attempt + 1)
            LOGGER.warning(
                "Gemini respondió %s; reintento %s/%s en %s segundos.",
                getattr(error, "code", "desconocido"),
                attempt + 2,
                GEMINI_MAX_ATTEMPTS,
                delay,
            )
            await asyncio.sleep(delay)


async def _gemini_synthesis(prompt: str) -> tuple[str, dict]:
    """Calls the official asynchronous google-genai client."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = await _generate_gemini_with_retry(
        client.aio.models.generate_content,
        model=GEMINI_MODEL,
        contents=prompt,
        config=_gemini_config(types),
    )
    return str(response.text), _gemini_usage(response, GEMINI_MODEL)


async def optionally_improve_answer(query: str, answer: str, enabled: bool) -> tuple[str, dict | None]:
    """Uses the configured production LLM only when explicitly enabled."""
    if not enabled:
        return answer, None

    prompt = _synthesis_prompt(query, answer)
    if API_LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            return answer, None
        return await _openai_synthesis(prompt)

    if API_LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            return answer, None
        return await _gemini_synthesis(prompt)

    raise ValueError(f"Proveedor de síntesis no soportado: {API_LLM_PROVIDER}")
