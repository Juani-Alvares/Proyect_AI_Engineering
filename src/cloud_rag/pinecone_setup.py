"""Creación o reutilización del índice Serverless de Pinecone."""

from pinecone import Pinecone, ServerlessSpec

from ..config import (
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
)

DIMENSION = 1536
METRIC = "cosine"


def create_or_get_index():
    """Devuelve el índice configurado y lo crea solo si no existe."""
    if not PINECONE_API_KEY:
        raise ValueError("Falta PINECONE_API_KEY en .env.")

    client = Pinecone(api_key=PINECONE_API_KEY)
    indexes = client.list_indexes()
    names = indexes.names() if hasattr(indexes, "names") else [item["name"] for item in indexes]

    if PINECONE_INDEX_NAME not in names:
        client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    return client.Index(PINECONE_INDEX_NAME)
