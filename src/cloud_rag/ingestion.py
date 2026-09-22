"""Ingesta de documentos técnicos en Pinecone."""

import hashlib

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import EMBEDDING_MODEL, PINECONE_NAMESPACE, PROJECT_ROOT
from .pinecone_setup import create_or_get_index

CHUNK_SIZE = 600
CHUNK_OVERLAP = 75


def category(source: str) -> str:
    if "database" in source:
        return "database"
    if "monitoring" in source:
        return "monitoring"
    return "api"


def load_chunks() -> list[Document]:
    """Lee data/ y crea chunks con metadata reutilizable por ambos retrievers."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    documents: list[Document] = []

    for path in sorted((PROJECT_ROOT / "data").iterdir()):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        for number, text in enumerate(splitter.split_text(path.read_text(encoding="utf-8"))):
            chunk_id = hashlib.sha256(f"{path.name}:{number}".encode()).hexdigest()
            metadata = {
                "source": path.name,
                "page": 1,
                "category": category(path.name),
                "chunk_id": chunk_id,
                "text": text,
            }
            documents.append(Document(page_content=text, metadata=metadata))

    return documents


def ingest_documents() -> int:
    """Genera embeddings y hace upsert con IDs deterministas en el namespace."""
    index = create_or_get_index()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    documents = load_chunks()
    texts = [document.page_content for document in documents]
    embeddings_list = embeddings.embed_documents(texts)
    vectors = [
        {
            "id": document.metadata["chunk_id"],
            "values": embedding,
            "metadata": document.metadata,
        }
        for document, embedding in zip(documents, embeddings_list, strict=True)
    ]

    if vectors:
        index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)

    return len(vectors)


if __name__ == "__main__":
    print(f"Chunks indexados: {ingest_documents()}")
