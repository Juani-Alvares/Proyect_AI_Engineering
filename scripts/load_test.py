"""Minimal no-dependency load test: five concurrent API tasks and p95 latency."""

import asyncio
import math
import os
import time

import httpx

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
QUERIES = [
    "Investiga el máximo de conexiones PostgreSQL y calcula el uso con 15 conexiones activas."
    for _ in range(5)
]


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


async def submit_and_wait(client: httpx.AsyncClient, query: str) -> float:
    start = time.perf_counter()
    response = await client.post("/tasks", json={"query": query})
    response.raise_for_status()
    job_id = response.json()["job_id"]
    while True:
        status_response = await client.get(f"/tasks/{job_id}")
        status_response.raise_for_status()
        task = status_response.json()
        if task["status"] in {"DONE", "FAILED", "REJECTED"}:
            if task["status"] != "DONE":
                raise RuntimeError(f"Trabajo {job_id} terminó como {task['status']}")
            return time.perf_counter() - start
        await asyncio.sleep(0.1)


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        latencies = await asyncio.gather(*(submit_and_wait(client, query) for query in QUERIES))
    print(f"Trabajos completados: {len(latencies)}")
    print(f"Latencia p95: {percentile_95(latencies):.3f} s")


if __name__ == "__main__":
    asyncio.run(main())
