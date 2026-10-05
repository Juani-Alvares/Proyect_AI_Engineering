"""Provider-selection tests for optional LLM synthesis without external calls."""

import asyncio

import pytest
from google.genai import errors

from app import llm


def test_disabled_llm_does_not_call_any_provider(monkeypatch):
    async def should_not_run(prompt):
        raise AssertionError("No provider should be called when USE_LLM is false")

    monkeypatch.setattr(llm, "_openai_synthesis", should_not_run)
    monkeypatch.setattr(llm, "_gemini_synthesis", should_not_run)

    result = asyncio.run(llm.optionally_improve_answer("consulta", "respuesta local", False))
    assert result == ("respuesta local", None)


def test_openai_provider_uses_openai_branch(monkeypatch):
    async def fake_openai(prompt):
        assert "consulta" in prompt
        return "síntesis OpenAI", {"model": "gpt-4o-mini"}

    monkeypatch.setattr(llm, "API_LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_openai_synthesis", fake_openai)

    result = asyncio.run(llm.optionally_improve_answer("consulta", "respuesta", True))
    assert result == ("síntesis OpenAI", {"model": "gpt-4o-mini"})


def test_gemini_provider_uses_gemini_branch(monkeypatch):
    async def fake_gemini(prompt):
        assert "respuesta" in prompt
        return "síntesis Gemini", {"model": "gemini-3.8-flash", "total_tokens": 12}

    monkeypatch.setattr(llm, "API_LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_gemini_synthesis", fake_gemini)

    result = asyncio.run(llm.optionally_improve_answer("consulta", "respuesta", True))
    assert result[0] == "síntesis Gemini"
    assert result[1]["total_tokens"] == 12


def test_gemini_failure_propagates_to_the_existing_worker_error_handling(monkeypatch):
    async def failing_gemini(prompt):
        raise RuntimeError("fallo Gemini controlado")

    monkeypatch.setattr(llm, "API_LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_gemini_synthesis", failing_gemini)

    with pytest.raises(RuntimeError, match="fallo Gemini controlado"):
        asyncio.run(llm.optionally_improve_answer("consulta", "respuesta", True))


def test_gemini_config_uses_low_thinking_and_enough_output_tokens():
    from google.genai import types

    config = llm._gemini_config(types)
    assert config.thinking_config.thinking_level is types.ThinkingLevel.LOW
    assert config.max_output_tokens >= 512


def test_gemini_usage_comes_from_response_usage_metadata():
    class Usage:
        prompt_token_count = 97
        candidates_token_count = 3
        total_token_count = 293

    class Response:
        usage_metadata = Usage()

    usage = llm._gemini_usage(Response(), "gemini-3.8-flash")
    assert usage == {
        "model": "gemini-3.8-flash",
        "input_tokens": 97,
        "output_tokens": 3,
        "total_tokens": 293,
    }


def test_gemini_retries_twice_then_returns_the_third_real_response(monkeypatch):
    response = object()
    outcomes = [
        errors.ServerError(503, {"error": "busy"}),
        errors.ServerError(503, {"error": "busy"}),
        response,
    ]
    delays = []

    async def generate_content(**kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)
    assert asyncio.run(llm._generate_gemini_with_retry(generate_content, model="test")) is response
    assert delays == [2, 4]


def test_gemini_propagates_after_three_transient_failures(monkeypatch):
    attempts = []
    delays = []

    async def generate_content(**kwargs):
        attempts.append(kwargs)
        raise errors.ServerError(503, {"error": "busy"})

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)
    with pytest.raises(errors.ServerError):
        asyncio.run(llm._generate_gemini_with_retry(generate_content, model="test"))
    assert len(attempts) == 3
    assert delays == [2, 4]


@pytest.mark.parametrize("status_code", [401, 404])
def test_gemini_does_not_retry_permanent_client_errors(monkeypatch, status_code):
    attempts = []

    async def generate_content(**kwargs):
        attempts.append(kwargs)
        raise errors.ClientError(status_code, {"error": "permanent"})

    async def should_not_sleep(seconds):
        raise AssertionError("Permanent errors must not be retried")

    monkeypatch.setattr(llm.asyncio, "sleep", should_not_sleep)
    with pytest.raises(errors.ClientError):
        asyncio.run(llm._generate_gemini_with_retry(generate_content, model="test"))
    assert len(attempts) == 1
