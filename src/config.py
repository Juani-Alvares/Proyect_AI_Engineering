"""Centralized environment configuration for the project."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "300"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "technical-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "technical-docs")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"
API_LLM_PROVIDER = os.getenv("API_LLM_PROVIDER", "openai").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "multi-agent-api")

_persist_dir = Path(os.getenv("CHROMA_PERSIST_DIR", "vectorstore"))
CHROMA_PERSIST_DIR = (
    _persist_dir if _persist_dir.is_absolute() else PROJECT_ROOT / _persist_dir
)
