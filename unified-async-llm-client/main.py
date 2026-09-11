"""Asynchronous example for the technical entity extraction pipeline."""

import asyncio
import logging

from dotenv import load_dotenv

from src.pipeline.chain import process_text


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    text = (
        "La API está desarrollada con FastAPI, utiliza Redis como caché y "
        "PostgreSQL como base de datos. Se detectaron problemas de conexiones "
        "concurrentes y aumento de latencia."
    )

    try:
        result = await process_text(text)
        print("Resultado validado:")
        print(result.model_dump_json(indent=2))
    except Exception as error:
        print(f"No fue posible procesar el texto: {error}")


if __name__ == "__main__":
    asyncio.run(main())
