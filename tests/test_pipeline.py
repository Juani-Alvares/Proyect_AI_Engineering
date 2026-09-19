import asyncio

from langchain_core.runnables import RunnableLambda

from src.pipeline import chain as chain_module
from src.schemas import TechnicalExtraction


class FakeChain:
    def __init__(self) -> None:
        self.received_input: dict[str, str] | None = None

    async def ainvoke(self, input_data: dict[str, str]) -> TechnicalExtraction:
        self.received_input = input_data
        return TechnicalExtraction(
            tecnologias=["FastAPI", "Redis", "PostgreSQL"],
            nivel_de_criticidad="alta",
            resumen_tecnico="Hay latencia y problemas de conexiones concurrentes.",
        )


def test_process_text_returns_valid_structure_with_mock(monkeypatch) -> None:
    fake_chain = FakeChain()
    monkeypatch.setattr(chain_module, "build_chain", lambda: fake_chain)

    result = asyncio.run(chain_module.process_text("Texto técnico de prueba"))

    assert isinstance(result, TechnicalExtraction)
    assert result.nivel_de_criticidad.value == "alta"
    assert fake_chain.received_input == {"text": "Texto técnico de prueba"}


def test_lcel_retry_retries_an_async_runnable() -> None:
    attempts = {"count": 0}

    async def fail_once(_: str) -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ValueError("Error temporal")
        return "correcto"

    runnable = RunnableLambda(lambda _: "correcto", afunc=fail_once).with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=False,
    )

    result = asyncio.run(runnable.ainvoke("entrada"))

    assert result == "correcto"
    assert attempts["count"] == 2
