"""FastAPI entry point for durable multi-agent tasks."""

from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request, status
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from src.config import REDIS_URL, USE_LLM

from .graph import create_production_graph
from .observability import configure_observability, traced_operation
from .redis_state import RedisJobStore
from .schemas import ApprovalRequest, HealthResponse, JobRecord, JobStatus, TaskAccepted, TaskRequest
from .worker import TaskWorker

def create_app(*, redis_client=None, graph=None, enable_observability: bool = True) -> FastAPI:
    """Creates an injectable app so tests never need a real Redis server."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_observability(enable_observability)
        owned_redis = redis_client is None
        client = redis_client or redis.from_url(REDIS_URL, decode_responses=True)
        app.state.store = RedisJobStore(client)

        if graph is None:
            async with AsyncRedisSaver.from_conn_string(REDIS_URL) as saver:
                await saver.asetup()
                app.state.worker = TaskWorker(app.state.store, create_production_graph(saver), use_llm=USE_LLM)
                await app.state.worker.start()
                yield
                await app.state.worker.stop()
        else:
            app.state.worker = TaskWorker(app.state.store, graph)
            await app.state.worker.start()
            yield
            await app.state.worker.stop()
        if owned_redis:
            await client.aclose()

    app = FastAPI(title="Multi-Agent Production API", version="1.0", lifespan=lifespan)

    @app.post("/tasks", response_model=TaskAccepted, status_code=status.HTTP_202_ACCEPTED)
    async def create_task(payload: TaskRequest, request: Request) -> TaskAccepted:
        job_id = uuid4()
        job = JobRecord(job_id=job_id, query=payload.query, thread_id=str(job_id))
        with traced_operation("task", job_id=str(job_id)):
            await request.app.state.store.create_job(job)
            await request.app.state.worker.enqueue(job_id)
        return TaskAccepted(job_id=job_id, status=JobStatus.PENDING)

    @app.get("/tasks/{job_id}", response_model=JobRecord)
    async def get_task(job_id: UUID, request: Request) -> JobRecord:
        job = await request.app.state.store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
        return job

    @app.post("/tasks/{job_id}/approve", response_model=TaskAccepted, status_code=status.HTTP_202_ACCEPTED)
    async def approve_task(job_id: UUID, payload: ApprovalRequest, request: Request) -> TaskAccepted:
        job = await request.app.state.store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
        if job.status is not JobStatus.WAITING_APPROVAL:
            raise HTTPException(status_code=409, detail="El trabajo no está esperando aprobación.")
        await request.app.state.worker.enqueue(job_id, payload.model_dump())
        return TaskAccepted(job_id=job_id, status=JobStatus.RUNNING)

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        available = await request.app.state.store.ping()
        return HealthResponse(status="ok" if available else "degraded", redis="ok" if available else "unavailable")

    return app


app = create_app()
