"""LCEL chain for extracting technical entities from a paragraph."""

import json
import logging
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from ..schemas import TechnicalExtraction
from .prompt import technical_extraction_prompt

logger = logging.getLogger(__name__)


class RetryLogHandler(BaseCallbackHandler):
    """Reports errors that LangChain may retry."""

    def on_chain_error(self, error: BaseException, **kwargs: object) -> None:
        logger.warning(
            "Error durante el procesamiento; LangChain reintentará si quedan intentos: %s",
            error,
        )


def _create_model() -> ChatOpenAI | ChatAnthropic:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model_name = os.getenv("LLM_MODEL")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    if provider == "anthropic":
        return ChatAnthropic(
            model=model_name or "claude-sonnet-5",
            temperature=temperature,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "300")),
        )
    if provider == "openai":
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=temperature,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "300")),
        )
    raise ValueError("LLM_PROVIDER debe ser 'openai' o 'anthropic'.")


def build_chain() -> Runnable:
    """Build the LCEL chain with structured output and two attempts."""
    model = _create_model()
    structured_model = model.with_structured_output(TechnicalExtraction).with_config(
        callbacks=[RetryLogHandler()]
    )
    retrying_model = structured_model.with_retry(
        stop_after_attempt=2,
        retry_if_exception_type=(
            OutputParserException,
            ValidationError,
            json.JSONDecodeError,
        ),
    )
    return technical_extraction_prompt | retrying_model


async def process_text(text: str) -> TechnicalExtraction:
    """Process text asynchronously and return a validated Pydantic object."""
    if not text.strip():
        raise ValueError("El texto de entrada no puede estar vacío.")

    load_dotenv()
    logger.info("Inicio del procesamiento de texto técnico.")
    chain = build_chain()

    try:
        result = await chain.ainvoke({"text": text})
        validated_result = TechnicalExtraction.model_validate(result)
        logger.info("Procesamiento y validación correctos.")
        return validated_result
    except Exception as error:
        logger.error("Error al procesar el texto después de los reintentos: %s", error)
        raise
