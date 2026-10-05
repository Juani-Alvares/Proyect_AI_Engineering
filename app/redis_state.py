"""Persistencia mínima de estados de trabajo con redis.asyncio."""

import json
from typing import Any
from uuid import UUID

from .schemas import JobRecord, utc_now


class RedisJobStore:
    """Guarda un JobRecord serializado bajo la clave job:{uuid}."""

    def __init__(self, redis_client: Any) -> None:
        self.redis = redis_client

    @staticmethod
    def key(job_id: UUID | str) -> str:
        return f"job:{job_id}"

    async def create_job(self, job: JobRecord) -> None:
        await self.redis.set(self.key(job.job_id), job.model_dump_json())

    async def get_job(self, job_id: UUID | str) -> JobRecord | None:
        value = await self.redis.get(self.key(job_id))
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return JobRecord.model_validate_json(value)

    async def update_job(self, job_id: UUID | str, **changes: Any) -> JobRecord | None:
        job = await self.get_job(job_id)
        if job is None:
            return None
        updated = job.model_copy(update={**changes, "updated_at": utc_now()})
        await self.redis.set(self.key(job_id), updated.model_dump_json())
        return updated

    async def ping(self) -> bool:
        try:
            return bool(await self.redis.ping())
        except Exception:
            return False
