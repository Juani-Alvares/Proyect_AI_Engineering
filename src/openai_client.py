"""Asynchronous OpenAI implementation."""

from collections.abc import AsyncIterator, Sequence

from openai import APIConnectionError, APIError, AsyncOpenAI, AuthenticationError, RateLimitError

from .base_client import BaseLLMClient
from .schemas import ChatMessage, ModelConfig, ModelResponse


class OpenAIClient(BaseLLMClient):
    provider_name = "openai"

    def __init__(self, api_key: str | None) -> None:
        self.client = AsyncOpenAI(api_key=api_key) if api_key else None

    @staticmethod
    def _error_message(error: Exception) -> str:
        if isinstance(error, AuthenticationError):
            return "La API key de OpenAI es inválida o no tiene permisos."
        if isinstance(error, RateLimitError):
            return "OpenAI alcanzó un límite de solicitudes o cuota. Intenta nuevamente más tarde."
        if isinstance(error, APIConnectionError):
            return "No fue posible conectarse con OpenAI. Revisa tu conexión a Internet."
        if isinstance(error, APIError):
            return f"OpenAI devolvió un error: {error}"
        return f"Error inesperado al usar OpenAI: {error}"

    def _missing_key_response(self, config: ModelConfig) -> ModelResponse:
        return ModelResponse(
            model=config.model,
            provider=self.provider_name,
            error="Falta OPENAI_API_KEY. Configúrala en el archivo .env.",
        )

    async def generate(
        self, messages: Sequence[ChatMessage], config: ModelConfig
    ) -> ModelResponse:
        if self.client is None:
            return self._missing_key_response(config)

        try:
            completion = await self.client.chat.completions.create(
                model=config.model,
                messages=[message.model_dump() for message in messages],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            content = completion.choices[0].message.content or ""
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
            yield "[Error: falta OPENAI_API_KEY. Configúrala en el archivo .env.]"
            return

        try:
            stream = await self.client.chat.completions.create(
                model=config.model,
                messages=[message.model_dump() for message in messages],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as error:
            yield f"[Error: {self._error_message(error)}]"

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
