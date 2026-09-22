"""Tests sin APIs reales para la Pre-entrega 4."""

import asyncio

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from evaluation.evaluate import calculate_metrics
from src.cloud_rag import ingestion
from src.cloud_rag.ingestion import CHUNK_OVERLAP, CHUNK_SIZE, category
from src.cloud_rag.pinecone_setup import DIMENSION, METRIC, create_or_get_index
from src.cloud_rag.retriever import RAGSystem


class FakeEmbeddings:
    def embed_query(self, _query):
        return [0.1, 0.2]


class FakeIndex:
    def __init__(self, matches):
        self.matches = matches
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return {"matches": self.matches}


class FakePineconeClient:
    created = None

    def __init__(self, api_key):
        self.api_key = api_key

    def list_indexes(self):
        return []

    def create_index(self, **kwargs):
        FakePineconeClient.created = kwargs

    def Index(self, name):
        return {"index_name": name}


class FakeBatchEmbeddings:
    def __init__(self):
        self.texts = []

    def embed_documents(self, texts):
        self.texts = list(texts)
        return [[float(position)] for position, _ in enumerate(texts)]


class FakeUpsertIndex:
    def __init__(self):
        self.vectors = []
        self.namespace = ""

    def upsert(self, *, vectors, namespace):
        self.vectors = vectors
        self.namespace = namespace


def document(source: str, chunk_id: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "page": 1,
            "category": "database",
            "chunk_id": chunk_id,
            "text": text,
        },
    )


def test_cloud_chunking_and_categories_are_configured():
    assert (CHUNK_SIZE, CHUNK_OVERLAP) == (600, 75)
    assert category("api_stack.md") == "api"
    assert category("database_connections.txt") == "database"
    assert category("monitoring.md") == "monitoring"


def test_ingestion_generates_embeddings_in_a_single_batch(monkeypatch):
    documents = [
        document("database_connections.txt", "db-1", "PostgreSQL pool"),
        document("api_stack.md", "api-1", "FastAPI endpoints"),
    ]
    fake_embeddings = FakeBatchEmbeddings()
    fake_index = FakeUpsertIndex()
    monkeypatch.setattr(ingestion, "load_chunks", lambda: documents)
    monkeypatch.setattr(ingestion, "create_or_get_index", lambda: fake_index)
    monkeypatch.setattr(ingestion, "OpenAIEmbeddings", lambda model: fake_embeddings)

    count = ingestion.ingest_documents()

    assert count == 2
    assert fake_embeddings.texts == ["PostgreSQL pool", "FastAPI endpoints"]
    assert [vector["values"] for vector in fake_index.vectors] == [[0.0], [1.0]]
    assert fake_index.namespace == ingestion.PINECONE_NAMESPACE


def test_pinecone_serverless_index_uses_expected_dimension_and_metric(monkeypatch):
    monkeypatch.setattr("src.cloud_rag.pinecone_setup.Pinecone", FakePineconeClient)
    monkeypatch.setattr("src.cloud_rag.pinecone_setup.PINECONE_API_KEY", "test-key")

    index = create_or_get_index()

    assert index["index_name"] == "technical-rag"
    assert DIMENSION == 1536
    assert METRIC == "cosine"
    assert FakePineconeClient.created["dimension"] == 1536
    assert FakePineconeClient.created["metric"] == "cosine"


def test_rag_system_returns_at_most_top_five_and_uses_namespace(monkeypatch):
    documents = [
        document("database_connections.txt", "db-1", "PostgreSQL pool 20 connections"),
        document("api_stack.md", "api-1", "FastAPI endpoints"),
        document("monitoring.md", "mon-1", "latency alerts"),
    ]
    index = FakeIndex([{"metadata": documents[0].metadata}, {"metadata": documents[1].metadata}])
    monkeypatch.setattr("src.cloud_rag.retriever.PINECONE_NAMESPACE", "technical-docs")

    system = RAGSystem(documents, index=index, embeddings=FakeEmbeddings())
    results = asyncio.run(system.retrieve("pool", k=5))

    assert len(results) <= 5
    assert results[0].metadata["source"] == "database_connections.txt"
    assert isinstance(system.ensemble, EnsembleRetriever)
    assert system.ensemble.weights == [0.6, 0.4]
    assert index.calls[0]["namespace"] == "technical-docs"
    assert index.calls[0]["top_k"] == 5


def test_bm25_returns_the_document_with_matching_technical_terms():
    documents = [
        document("database_connections.txt", "db-1", "PostgreSQL pool de 20 conexiones"),
        document("api_stack.md", "api-1", "Endpoints implementados con FastAPI"),
    ]
    system = RAGSystem(documents, index=FakeIndex([]), embeddings=FakeEmbeddings())

    results = system.retrieve_lexical("postgresql conexiones", k=2)

    assert results[0].metadata["source"] == "database_connections.txt"
    assert results[0].metadata["chunk_id"] == "db-1"


def test_retrieved_documents_keep_required_metadata():
    item = document("database_connections.txt", "db-1", "PostgreSQL")
    assert set(item.metadata) == {"source", "page", "category", "chunk_id", "text"}


def test_precision_and_recall_at_five():
    precision, recall = calculate_metrics(
        ["api_stack.md", "database_connections.txt", "monitoring.md"],
        "database_connections.txt",
    )
    assert precision == 0.2
    assert recall == 1.0

    precision, recall = calculate_metrics(["api_stack.md"], "database_connections.txt")
    assert precision == 0.0
    assert recall == 0.0
