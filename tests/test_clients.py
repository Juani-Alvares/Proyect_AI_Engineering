import asyncio

from src.anthropic_client import AnthropicClient
from src.openai_client import OpenAIClient
from src.schemas import ChatMessage, ModelConfig


def test_openai_missing_key_generate_and_stream() -> None:
    client = OpenAIClient(None)
    config = ModelConfig(model="gpt-4o-mini")
    messages = [ChatMessage(role="user", content="Hola")]

    response = asyncio.run(client.generate(messages, config))
    assert response.error is not None
    assert "OPENAI_API_KEY" in response.error

    async def collect():
        return [chunk async for chunk in client.stream(messages, config)]

    chunks = asyncio.run(collect())
    assert "OPENAI_API_KEY" in "".join(chunks)


def test_anthropic_missing_key_generate_and_stream() -> None:
    client = AnthropicClient(None)
    config = ModelConfig(model="claude-sonnet-5")
    messages = [ChatMessage(role="user", content="Hola")]

    response = asyncio.run(client.generate(messages, config))
    assert response.error is not None
    assert "ANTHROPIC_API_KEY" in response.error

    async def collect():
        return [chunk async for chunk in client.stream(messages, config)]

    chunks = asyncio.run(collect())
    assert "ANTHROPIC_API_KEY" in "".join(chunks)
