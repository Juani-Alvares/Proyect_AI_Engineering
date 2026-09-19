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

_persist_dir = Path(os.getenv("CHROMA_PERSIST_DIR", "vectorstore"))
CHROMA_PERSIST_DIR = (
    _persist_dir if _persist_dir.is_absolute() else PROJECT_ROOT / _persist_dir
)
