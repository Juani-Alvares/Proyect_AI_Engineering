"""Document ingestion and persistent Chroma vector store creation."""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import CHROMA_PERSIST_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = CHROMA_PERSIST_DIR
COLLECTION_NAME = "technical_documents"

# The splitter uses tiktoken so chunk size and overlap are measured in tokens.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Read supported local files and retain their source metadata."""
    documents: list[Document] = []
    for path in sorted(data_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            documents.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": path.name},
                )
            )
    logger.info("Documentos encontrados para ingesta: %s", len(documents))
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks and number each source chunk."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk"] = index
    logger.info("Chunks creados: %s", len(chunks))
    return chunks


def vectorstore_exists(vectorstore_dir: Path = VECTORSTORE_DIR) -> bool:
    """Return whether a persistent Chroma database has already been created."""
    return (vectorstore_dir / "chroma.sqlite3").exists()


def create_or_load_vectorstore(
    embeddings: Embeddings,
    data_dir: Path = DATA_DIR,
    vectorstore_dir: Path = VECTORSTORE_DIR,
) -> Chroma:
    """Create the local database once, then reuse it in later executions."""
    if vectorstore_exists(vectorstore_dir):
        logger.info("Reutilizando vectorstore persistente: %s", vectorstore_dir)
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(vectorstore_dir),
        )

    logger.info("Inicio de ingesta para crear el vectorstore local.")
    documents = load_documents(data_dir)
    if not documents:
        raise ValueError("No se encontraron archivos .txt o .md en data/.")

    chunks = split_documents(documents)
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(vectorstore_dir),
    )
    logger.info("Vectorstore persistente creado: %s", vectorstore_dir)
    return vectorstore
