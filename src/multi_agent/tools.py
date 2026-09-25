"""Local, deterministic tools used by the specialized agents."""

import re

from langchain_core.tools import tool

from ..config import PROJECT_ROOT

STOP_WORDS = {
    "a", "al", "con", "de", "del", "el", "en", "es", "la", "las", "lo",
    "los", "para", "por", "que", "qué", "respecto", "se", "un", "una", "y",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if token not in STOP_WORDS
    }


@tool
def search_technical_docs(query: str) -> str:
    """Search local technical documents and return the most relevant evidence."""
    query_tokens = _tokens(query)
    candidates: list[tuple[int, str, str]] = []

    for path in sorted((PROJECT_ROOT / "data").iterdir()):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue

        for sentence in re.split(r"(?<=[.!?])\s+", path.read_text(encoding="utf-8")):
            score = len(query_tokens & _tokens(sentence))
            if score:
                candidates.append((score, path.name, sentence.strip()))

    if not candidates:
        return "No se encontró evidencia relevante en los documentos técnicos locales."

    _, source, evidence = max(candidates, key=lambda item: (item[0], item[1]))
    return f"Fuente: {source}\nEvidencia: {evidence}"


@tool
def calculate_percentage(part: float, total: float) -> str:
    """Calculate what percentage part represents of total."""
    if total <= 0:
        return "No se puede calcular un porcentaje con un total menor o igual a cero."

    percentage = part / total * 100
    return f"{part:g} representa {percentage:g}% de {total:g}."
