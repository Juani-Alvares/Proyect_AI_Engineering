"""Similarity retrieval and context formatting for the RAG pipeline."""

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)
TOP_K = 3


def retrieve_documents(vectorstore: Chroma, query: str, k: int = TOP_K) -> list[Document]:
    """Retrieve only the most relevant document chunks."""
    logger.info("Inicio de recuperación semántica.")
    documents = vectorstore.similarity_search(query, k=k)
    logger.info("Fragmentos recuperados: %s", len(documents))
    return documents


def format_documents(documents: list[Document]) -> str:
    """Format retrieved chunks with source labels for the grounded prompt."""
    return "\n\n".join(
        f"Fuente: {document.metadata.get('source', 'desconocida')}\n{document.page_content}"
        for document in documents
    )
