"""Schemas HTTP y de persistencia para los trabajos asíncronos."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


def utc_now() -> datetime:
    return datetime.now(UTC)


class TaskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4_000)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("La consulta no puede estar vacía.")
        return value


class ApprovalRequest(BaseModel):
    approved: bool
    comment: str = Field(default="", max_length=1_000)


class JobRecord(BaseModel):
    job_id: UUID
    query: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    result: dict[str, Any] | None = None
    error: str | None = None
    thread_id: str


class TaskAccepted(BaseModel):
    job_id: UUID
    status: JobStatus


class HealthResponse(BaseModel):
    status: str
    redis: str
