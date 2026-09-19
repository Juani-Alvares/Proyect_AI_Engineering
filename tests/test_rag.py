import asyncio

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.rag import chain as rag_chain
from src.rag import ingestion
from src.rag.ingestion import create_or_load_vectorstore, vectorstore_exists
from src.schemas import RAGResponse


class FakeVectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.last_query = ""

    def similarity_search(self, query: str, k: int) -> list[Document]:
        self.last_query = query
        return self.documents[:k]


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


def configure_fake_rag(monkeypatch, documents: list[Document], response: str) -> FakeVectorStore:
    vectorstore = FakeVectorStore(documents)
    monkeypatch.setattr(rag_chain, "create_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(
        rag_chain,
        "create_or_load_vectorstore",
        lambda _: vectorstore,
    )

    async def answer(_: object) -> str:
        return response

    monkeypatch.setattr(
        rag_chain,
        "_create_rag_model",
        lambda: RunnableLambda(lambda _: response, afunc=answer),
    )
    return vectorstore


def test_known_question_returns_grounded_response(monkeypatch) -> None:
    documents = [
        Document(
            page_content="El pool de PostgreSQL tiene un máximo de 20 conexiones.",
            metadata={"source": "database_connections.txt"},
        )
    ]
    vectorstore = configure_fake_rag(
        monkeypatch,
        documents,
        '{"respuesta": "El máximo es 20 conexiones.", "referencias": ["database_connections.txt"]}',
    )

    result = asyncio.run(
        rag_chain.get_rag_response(
            "¿Cuál es el máximo de conexiones del pool de PostgreSQL?"
        )
    )

    assert isinstance(result, RAGResponse)
    assert result.respuesta
    assert result.referencias == ["database_connections.txt"]
    assert vectorstore.last_query


def test_unknown_question_returns_no_lo_se(monkeypatch) -> None:
    documents = [
        Document(
            page_content="Redis se usa como caché.",
            metadata={"source": "api_stack.md"},
        )
    ]
    configure_fake_rag(
        monkeypatch,
        documents,
        '{"respuesta": "No lo sé.", "referencias": []}',
    )

    result = asyncio.run(
        rag_chain.get_rag_response("¿Qué proveedor de pagos utiliza la API?")
    )

    assert result.respuesta == "No lo sé."
    assert result.referencias == []


def test_ingestion_creates_and_reuses_persistent_chroma(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    vectorstore_dir = tmp_path / "vectorstore"
    data_dir.mkdir()
    (data_dir / "example.md").write_text(
        "FastAPI utiliza PostgreSQL como base de datos.", encoding="utf-8"
    )
    embeddings = FakeEmbeddings()

    class FakeSplitter:
        def split_documents(self, documents: list[Document]) -> list[Document]:
            return documents

    def fake_from_tiktoken_encoder(**kwargs: int) -> FakeSplitter:
        assert kwargs == {"chunk_size": 500, "chunk_overlap": 50}
        return FakeSplitter()

    monkeypatch.setattr(
        ingestion.RecursiveCharacterTextSplitter,
        "from_tiktoken_encoder",
        fake_from_tiktoken_encoder,
    )

    create_or_load_vectorstore(embeddings, data_dir, vectorstore_dir)
    assert vectorstore_exists(vectorstore_dir)

    reused_store = create_or_load_vectorstore(embeddings, data_dir, vectorstore_dir)
    assert reused_store is not None
