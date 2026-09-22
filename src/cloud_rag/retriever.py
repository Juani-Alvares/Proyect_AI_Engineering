"""Recuperación híbrida con Pinecone, BM25 y EnsembleRetriever."""

import re
from typing import Any

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from pydantic import PrivateAttr
from rank_bm25 import BM25Okapi

from ..config import EMBEDDING_MODEL, PINECONE_NAMESPACE
from .pinecone_setup import create_or_get_index

VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
TOP_K = 5


class RankBM25Retriever(BaseRetriever):
    """Retriever léxico local basado en rank-bm25."""

    documents: list[Document]
    k: int = TOP_K
    _bm25: BM25Okapi = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._bm25 = BM25Okapi([self._tokenize(document.page_content) for document in self.documents])

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        scores = self._bm25.get_scores(self._tokenize(query))
        positions = sorted(range(len(self.documents)), key=lambda position: (-scores[position], position))
        return [self.documents[position] for position in positions[: self.k]]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class PineconeVectorRetriever(BaseRetriever):
    """Retriever vectorial que adapta la consulta síncrona de Pinecone."""

    index: Any
    embeddings: Any
    namespace: str
    k: int = TOP_K

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        response = self.index.query(
            vector=self.embeddings.embed_query(query),
            top_k=self.k,
            namespace=self.namespace,
            include_metadata=True,
        )
        matches = response.matches if hasattr(response, "matches") else response["matches"]
        return [self._document_from_match(match) for match in matches]

    @staticmethod
    def _document_from_match(match) -> Document:
        metadata = match.metadata if hasattr(match, "metadata") else match["metadata"]
        return Document(page_content=metadata["text"], metadata=metadata)


class RAGSystem:
    """Encapsula un EnsembleRetriever para recuperar hasta cinco chunks."""

    def __init__(self, documents: list[Document], index=None, embeddings=None):
        self.documents = documents
        self.embeddings = embeddings or OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.index = index or create_or_get_index()
        self.bm25_retriever = RankBM25Retriever(documents=documents)
        self.vector_retriever = PineconeVectorRetriever(
            index=self.index,
            embeddings=self.embeddings,
            namespace=PINECONE_NAMESPACE,
        )
        self.ensemble = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[VECTOR_WEIGHT, BM25_WEIGHT],
            id_key="chunk_id",
        )

    async def retrieve(self, query: str, k: int = TOP_K) -> list[Document]:
        """Ejecuta el ensemble asíncrono y limita el resultado a top-k."""
        if not query.strip():
            raise ValueError("La consulta no puede estar vacía.")

        documents = await self.ensemble.ainvoke(query)
        return documents[:k]

    def retrieve_lexical(self, query: str, k: int = TOP_K) -> list[Document]:
        """Expone el ranking BM25 para pruebas y uso puntual."""
        return self.bm25_retriever.invoke(query)[:k]
