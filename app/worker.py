"""In-process asynchronous worker queue for long-running graph jobs."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langgraph.types import Command

from .graph import create_production_state
from .llm import optionally_improve_answer
from .observability import traced_operation
from .redis_state import RedisJobStore
from .schemas import JobStatus

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkItem:
    job_id: UUID
    approval: dict[str, Any] | None = None


class TaskWorker:
    """Consumes jobs without blocking FastAPI request handlers."""

    def __init__(self, store: RedisJobStore, graph: Any, use_llm: bool = False) -> None:
        self.store = store
        self.graph = graph
        self.use_llm = use_llm
        self.queue: asyncio.Queue[WorkItem] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []

    async def start(self, count: int = 1) -> None:
        self._tasks = [asyncio.create_task(self._consume(), name=f"task-worker-{index}") for index in range(count)]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def enqueue(self, job_id: UUID, approval: dict[str, Any] | None = None) -> None:
        await self.queue.put(WorkItem(job_id, approval))

    async def _consume(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                await self.process(item)
            finally:
                self.queue.task_done()

    async def process(self, item: WorkItem) -> None:
        job = await self.store.get_job(item.job_id)
        if job is None:
            return
        await self.store.update_job(item.job_id, status=JobStatus.RUNNING, error=None)
        config = {"configurable": {"thread_id": job.thread_id}}
        try:
            with traced_operation("worker", job_id=str(item.job_id)):
                with traced_operation("graph", thread_id=job.thread_id):
                    if item.approval is None:
                        result = await self.graph.ainvoke(create_production_state(job.query), config=config)
                    else:
                        result = await self.graph.ainvoke(Command(resume=item.approval), config=config)

                if "__interrupt__" in result:
                    await self.store.update_job(
                        item.job_id,
                        status=JobStatus.WAITING_APPROVAL,
                        result={"requires_approval": True, "message": "Esperando decisión humana."},
                    )
                    return
                if not result.get("rejected"):
                    with traced_operation("llm_synthesis", job_id=str(item.job_id)):
                        improved_answer, usage = await optionally_improve_answer(
                            job.query, str(result.get("final_answer", "")), self.use_llm
                        )
                    result = {**result, "final_answer": improved_answer}
                    if usage:
                        result["llm_usage"] = usage
            status = JobStatus.REJECTED if result.get("rejected") else JobStatus.DONE
            await self.store.update_job(item.job_id, status=status, result=self._public_result(result))
        except Exception as error:
            LOGGER.exception("Falló el trabajo %s", item.job_id)
            await self.store.update_job(item.job_id, status=JobStatus.FAILED, error=str(error))

    @staticmethod
    def _public_result(result: dict[str, Any]) -> dict[str, Any]:
        """Persists only JSON-friendly fields relevant to the API consumer."""
        keys = (
            "final_answer",
            "research_result",
            "analysis_result",
            "validation_result",
            "execution_trace",
            "task_completed",
            "llm_usage",
        )
        return {key: result[key] for key in keys if key in result}
