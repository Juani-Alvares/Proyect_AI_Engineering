"""Asynchronous LCEL chain for local semantic retrieval."""

import asyncio
import logging
import os

from langchain_core.runnables import Runnable
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ..schemas import RAGResponse
from .ingestion import create_or_load_vectorstore
from .prompt import rag_output_parser, rag_prompt
from .retriever import format_documents, retrieve_documents

logger = logging.getLogger(__name__)


def create_embeddings() -> OpenAIEmbeddings:
    """Create the single embedding model used for indexing and searching."""
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )


def _create_rag_model() -> ChatOpenAI | ChatAnthropic:
    """Reuse the existing OpenAI configuration for grounded answers."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model_name = os.getenv("LLM_MODEL")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    if provider == "anthropic":
        return ChatAnthropic(
            model=model_name or "claude-sonnet-5",
            temperature=temperature,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "300")),
        )
    if provider != "openai":
        raise ValueError("LLM_PROVIDER debe ser 'openai' o 'anthropic'.")
    return ChatOpenAI(
        model=model_name or "gpt-4o-mini",
        temperature=temperature,
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "300")),
    )


def build_rag_chain(model: Runnable) -> Runnable:
    """Create Prompt -> LLM -> PydanticOutputParser using LCEL."""
    return rag_prompt | model | rag_output_parser


async def get_rag_response(query: str) -> RAGResponse:
    """Retrieve relevant chunks and asynchronously generate a grounded answer."""
    if not query.strip():
        raise ValueError("La consulta no puede estar vacía.")

    embeddings = create_embeddings()
    vectorstore = create_or_load_vectorstore(embeddings)
    documents = await asyncio.to_thread(retrieve_documents, vectorstore, query)
    context = format_documents(documents)
    logger.info("Inicio de generación de respuesta RAG.")

    try:
        chain = build_rag_chain(_create_rag_model())
        result = await chain.ainvoke({"question": query, "context": context})
        logger.info("Respuesta RAG generada y validada.")
        return result
    except Exception as error:
        logger.error("Error al generar la respuesta RAG: %s", error)
        raise
