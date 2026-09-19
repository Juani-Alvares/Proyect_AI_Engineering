"""Asynchronous Anthropic implementation."""

from collections.abc import AsyncIterator, Sequence

from anthropic import APIConnectionError, APIError, AsyncAnthropic, AuthenticationError, RateLimitError

from .base_client import BaseLLMClient
from .schemas import ChatMessage, ModelConfig, ModelResponse


class AnthropicClient(BaseLLMClient):
    provider_name = "anthropic"

    def __init__(self, api_key: str | None) -> None:
        self.client = AsyncAnthropic(api_key=api_key) if api_key else None

    @staticmethod
    def _error_message(error: Exception) -> str:
        if isinstance(error, AuthenticationError):
            return "La API key de Anthropic es inválida o no tiene permisos."
        if isinstance(error, RateLimitError):
            return "Anthropic alcanzó un límite de solicitudes o cuota. Intenta nuevamente más tarde."
        if isinstance(error, APIConnectionError):
            return "No fue posible conectarse con Anthropic. Revisa tu conexión a Internet."
        if isinstance(error, APIError):
            return f"Anthropic devolvió un error: {error}"
        return f"Error inesperado al usar Anthropic: {error}"

    @staticmethod
    def _convert_messages(messages: Sequence[ChatMessage]) -> tuple[str | None, list[dict[str, str]]]:
        system_parts = [message.content for message in messages if message.role == "system"]
        anthropic_messages = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role != "system"
        ]
        return "\n\n".join(system_parts) or None, anthropic_messages

    def _missing_key_response(self, config: ModelConfig) -> ModelResponse:
        return ModelResponse(
            model=config.model,
            provider=self.provider_name,
            error="Falta ANTHROPIC_API_KEY. Configúrala en el archivo .env.",
        )

    async def generate(
        self, messages: Sequence[ChatMessage], config: ModelConfig
    ) -> ModelResponse:
        if self.client is None:
            return self._missing_key_response(config)

        system, anthropic_messages = self._convert_messages(messages)
        request = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": anthropic_messages,
        }
        if system is not None:
            request["system"] = system
        try:
            response = await self.client.messages.create(**request)
            content = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return ModelResponse(
                content=content, model=config.model, provider=self.provider_name
            )
        except Exception as error:
            return ModelResponse(
                model=config.model,
                provider=self.provider_name,
                error=self._error_message(error),
            )

    async def stream(
        self, messages: Sequence[ChatMessage], config: ModelConfig
    ) -> AsyncIterator[str]:
        if self.client is None:
            yield "[Error: falta ANTHROPIC_API_KEY. Configúrala en el archivo .env.]"
            return

        system, anthropic_messages = self._convert_messages(messages)
        request = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": anthropic_messages,
        }
        if system is not None:
            request["system"] = system
        try:
            async with self.client.messages.stream(**request) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as error:
            yield f"[Error: {self._error_message(error)}]"

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
