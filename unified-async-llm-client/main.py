"""Small asynchronous validation script for the unified client."""

import asyncio
import os

from dotenv import load_dotenv

from src.manager import AsyncLLMManager
from src.schemas import ChatMessage, ModelConfig


def default_model(provider: str) -> str:
    if provider == "anthropic":
        return "claude-sonnet-5"
    return "gpt-4o-mini"


async def main() -> None:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL") or default_model(provider)

    try:
        config = ModelConfig(
            model=model,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "300")),
        )
        manager = AsyncLLMManager(provider)
    except (ValueError, TypeError) as error:
        print(f"Error de configuración: {error}")
        return

    messages = [ChatMessage(role="user", content="¿Qué es la entropía?")]

    try:
        print(f"Proveedor: {provider} | Modelo: {config.model}\n")
        print("Respuesta normal:")
        response = await manager.generate(messages, config)
        if response.error:
            print(f"Error: {response.error}")
        else:
            print(response.content)

        print("\nStreaming:")
        async for chunk in manager.stream(messages, config):
            print(chunk, end="", flush=True)
        print()
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
