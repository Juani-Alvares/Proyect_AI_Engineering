"""Worker tests use graph doubles instead of paid APIs or a Redis server."""

import asyncio
from uuid import uuid4

from google import genai
from google.genai import errors

from app import llm
from app.redis_state import RedisJobStore
from app.schemas import JobRecord, JobStatus
from app.worker import TaskWorker, WorkItem


class FakeRedis:
    def __init__(self):
        self.data = {}

    async def set(self, key, value):
        self.data[key] = value

    async def get(self, key):
        return self.data.get(key)

    async def ping(self):
        return True


class WaitingGraph:
    async def ainvoke(self, state, config):
        if hasattr(state, "resume"):
            return {"final_answer": "Aprobada", "task_completed": True}
        return {"__interrupt__": [{"requires_approval": True}]}


class FailingGraph:
    async def ainvoke(self, state, config):
        raise RuntimeError("fallo controlado")


class CompleteGraph:
    async def ainvoke(self, state, config):
        return {"final_answer": "Respuesta del grafo", "task_completed": True}


def test_worker_waits_then_resumes_a_human_approval():
    async def scenario():
        store = RedisJobStore(FakeRedis())
        job = JobRecord(job_id=uuid4(), query="enviar pago", thread_id="thread")
        await store.create_job(job)
        worker = TaskWorker(store, WaitingGraph())

        await worker.process(WorkItem(job.job_id))
        assert (await store.get_job(job.job_id)).status is JobStatus.WAITING_APPROVAL

        await worker.process(WorkItem(job.job_id, {"approved": True, "comment": "ok"}))
        assert (await store.get_job(job.job_id)).status is JobStatus.DONE

    asyncio.run(scenario())


def test_worker_marks_failures_without_crashing():
    async def scenario():
        store = RedisJobStore(FakeRedis())
        job = JobRecord(job_id=uuid4(), query="consulta", thread_id="thread")
        await store.create_job(job)
        await TaskWorker(store, FailingGraph()).process(WorkItem(job.job_id))
        saved = await store.get_job(job.job_id)
        assert saved.status is JobStatus.FAILED
        assert "fallo controlado" in saved.error

    asyncio.run(scenario())


def test_worker_marks_a_gemini_synthesis_failure_as_failed(monkeypatch):
    async def failing_synthesis(query, answer, enabled):
        raise RuntimeError("fallo Gemini controlado")

    monkeypatch.setattr("app.worker.optionally_improve_answer", failing_synthesis)

    async def scenario():
        store = RedisJobStore(FakeRedis())
        job = JobRecord(job_id=uuid4(), query="consulta", thread_id="thread")
        await store.create_job(job)
        await TaskWorker(store, CompleteGraph(), use_llm=True).process(WorkItem(job.job_id))
        saved = await store.get_job(job.job_id)
        assert saved.status is JobStatus.FAILED
        assert "fallo Gemini controlado" in saved.error

    asyncio.run(scenario())


def test_worker_marks_done_after_a_transient_gemini_retry(monkeypatch):
    class Usage:
        prompt_token_count = 97
        candidates_token_count = 3
        total_token_count = 293

    class Response:
        text = "Síntesis completa"
        usage_metadata = Usage()

    class FakeModels:
        def __init__(self):
            self.calls = 0

        async def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise errors.ServerError(503, {"error": "busy"})
            return Response()

    class FakeClient:
        def __init__(self):
            self.aio = type("Aio", (), {"models": FakeModels()})()

    fake_client = FakeClient()
    delays = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(llm, "API_LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(genai, "Client", lambda api_key: fake_client)
    monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)

    async def scenario():
        store = RedisJobStore(FakeRedis())
        job = JobRecord(job_id=uuid4(), query="consulta", thread_id="thread")
        await store.create_job(job)
        await TaskWorker(store, CompleteGraph(), use_llm=True).process(WorkItem(job.job_id))
        saved = await store.get_job(job.job_id)
        assert saved.status is JobStatus.DONE
        assert saved.result["llm_usage"]["total_tokens"] == 293

    asyncio.run(scenario())
    assert fake_client.aio.models.calls == 2
    assert delays == [2]
