"""Provider selection behind a single asynchronous interface."""

from collections.abc import AsyncIterator, Sequence
import os

from .anthropic_client import AnthropicClient
from .base_client import BaseLLMClient
from .openai_client import OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse


class AsyncLLMManager:
    """Selects and uses one provider without exposing SDK differences."""

    def __init__(self, provider: str | None = None) -> None:
        selected_provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        if selected_provider == "openai":
            self.client: BaseLLMClient = OpenAIClient(os.getenv("OPENAI_API_KEY"))
        elif selected_provider == "anthropic":
            self.client = AnthropicClient(os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError("LLM_PROVIDER debe ser 'openai' o 'anthropic'.")

    async def generate(
        self, messages: Sequence[ChatMessage], config: ModelConfig
    ) -> ModelResponse:
        return await self.client.generate(messages, config)

    async def stream(
        self, messages: Sequence[ChatMessage], config: ModelConfig
    ) -> AsyncIterator[str]:
        async for chunk in self.client.stream(messages, config):
            yield chunk

    async def close(self) -> None:
        await self.client.close()
