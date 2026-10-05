"""The production graph retains the approved local multi-agent behaviour."""

import asyncio

from app.graph import create_production_graph, create_production_state


DEMO_QUERY = (
    "Investiga cuál es el máximo de conexiones configurado para PostgreSQL y calcula "
    "qué porcentaje representan 15 conexiones activas respecto del máximo."
)


def test_production_graph_completes_without_external_services():
    async def scenario():
        result = await create_production_graph().ainvoke(create_production_state(DEMO_QUERY))
        assert result["task_completed"] is True
        assert result["final_answer"]
        assert result["execution_trace"] == [
            "supervisor", "research", "supervisor", "analysis",
            "supervisor", "validation", "supervisor", "hitl", "finalize",
        ]

    asyncio.run(scenario())
