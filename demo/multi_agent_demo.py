"""Reproducible, API-free demonstration of the multi-agent graph."""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multi_agent.graph import build_multi_agent_graph
from src.multi_agent.state import create_initial_state

QUERY = (
    "Investiga cuál es el máximo de conexiones configurado para PostgreSQL y "
    "calcula qué porcentaje representan 15 conexiones activas respecto del máximo. "
    "Explica el resultado."
)


async def main() -> None:
    graph = build_multi_agent_graph()
    result = await graph.ainvoke(create_initial_state(QUERY))

    print(f"USER:\n{QUERY}\n")
    print("TRACE:")
    print(" -> ".join(result["execution_trace"]))
    print(f"\nRESEARCH:\n{result['research_result']}")
    print(f"\nANALYST:\n{result['analysis_result']}")
    print(f"\nVALIDATION:\n{result['validation_result']}")
    print(f"\nFINAL:\n{result['final_answer']}")
    print(f"\ntask_completed = {result['task_completed']}")


if __name__ == "__main__":
    asyncio.run(main())
