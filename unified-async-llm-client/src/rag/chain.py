"""Asynchronous LCEL chain for local semantic retrieval."""

import asyncio
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ..config import (
    EMBEDDING_MODEL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
)
from ..schemas import RAGResponse
from .ingestion import create_or_load_vectorstore
from .prompt import rag_output_parser, rag_prompt
from .retriever import format_documents, retrieve_documents

logger = logging.getLogger(__name__)


def create_embeddings() -> OpenAIEmbeddings:
    """Create the single embedding model used for indexing and searching."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _create_rag_model() -> ChatOpenAI | ChatAnthropic:
    """Reuse the existing OpenAI configuration for grounded answers."""
    if LLM_PROVIDER == "anthropic":
        return ChatAnthropic(
            model=LLM_MODEL or "claude-sonnet-5",
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    if LLM_PROVIDER != "openai":
        raise ValueError("LLM_PROVIDER debe ser 'openai' o 'anthropic'.")
    return ChatOpenAI(
        model=LLM_MODEL or "gpt-4o-mini",
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
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
