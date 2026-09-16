"""Asynchronous example for the local RAG system."""

import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()

from src.rag.chain import get_rag_response


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    questions = [
        "¿Cuál es el máximo de conexiones del pool de PostgreSQL?",
        "¿Qué proveedor de pagos utiliza la API?",
    ]

    for question in questions:
        try:
            result = await get_rag_response(question)
            print(f"\nPregunta: {question}")
            print(result.model_dump_json(indent=2))
        except Exception as error:
            print(f"No fue posible procesar la pregunta: {error}")


if __name__ == "__main__":
    asyncio.run(main())
