"""Abstract interface for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

from .schemas import ChatMessage, ModelConfig, ModelResponse


class BaseLLMClient(ABC):
    """Common asynchronous interface implemented by each provider."""

    provider_name: str

    @abstractmethod
    async def generate(
        self, messages: Sequence[ChatMessage], config: ModelConfig
    ) -> ModelResponse:
        """Generate a complete answer."""

    @abstractmethod
    async def stream(
        self, messages: Sequence[ChatMessage], config: ModelConfig
    ) -> AsyncIterator[str]:
        """Yield answer fragments as they arrive."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources used by the provider client."""
