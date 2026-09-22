"""Evaluación simple del recuperador cloud con un Golden Set."""

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cloud_rag.ingestion import load_chunks
from src.cloud_rag.retriever import RAGSystem

GOLDEN_SET_PATH = Path(__file__).with_name("golden_set.json")


def calculate_metrics(sources: list[str], expected_source: str, k: int = 5) -> tuple[float, float]:
    """Calcula Recall@k y Precision@k con un único documento relevante."""
    relevant = sum(source == expected_source for source in sources[:k])
    recall = 1.0 if relevant else 0.0
    precision = relevant / k
    return precision, recall


async def evaluate() -> None:
    golden_set = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    rag_system = RAGSystem(load_chunks())
    precisions: list[float] = []
    recalls: list[float] = []

    for number, item in enumerate(golden_set, start=1):
        results = await rag_system.retrieve(item["pregunta"], k=5)
        sources = [document.metadata["source"] for document in results]
        precision, recall = calculate_metrics(sources, item["documento_id_esperado"])
        precisions.append(precision)
        recalls.append(recall)

        print(f"Pregunta {number}: {item['pregunta']}")
        print(f"Esperado: {item['documento_id_esperado']}")
        print("Recuperados:")
        for position, source in enumerate(sources, start=1):
            print(f"{position}. {source}")
        print(f"Recall@5: {recall:.2f}")
        print(f"Precision@5: {precision:.2f}\n")

    print("RESULTADO FINAL")
    print(f"Mean Recall@5: {sum(recalls) / len(recalls):.2f}")
    print(f"Mean Precision@5: {sum(precisions) / len(precisions):.2f}")


if __name__ == "__main__":
    asyncio.run(evaluate())
