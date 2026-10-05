"""Tests for the Redis job persistence adapter."""

import asyncio
from uuid import uuid4

from app.redis_state import RedisJobStore
from app.schemas import JobRecord, JobStatus


class FakeRedis:
    def __init__(self):
        self.data = {}

    async def set(self, key, value):
        self.data[key] = value

    async def get(self, key):
        return self.data.get(key)

    async def ping(self):
        return True


def test_redis_job_store_creates_reads_and_updates_jobs():
    async def scenario():
        store = RedisJobStore(FakeRedis())
        job = JobRecord(job_id=uuid4(), query="consulta", thread_id="thread-1")
        await store.create_job(job)

        loaded = await store.get_job(job.job_id)
        assert loaded is not None
        assert loaded.status is JobStatus.PENDING

        updated = await store.update_job(job.job_id, status=JobStatus.RUNNING)
        assert updated is not None
        assert updated.status is JobStatus.RUNNING
        assert await store.ping() is True

    asyncio.run(scenario())
