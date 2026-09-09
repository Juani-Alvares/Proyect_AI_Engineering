"""Pydantic models used by the clients."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """A message sent to a language model."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content no puede estar vacío")
        return value


class ModelConfig(BaseModel):
    """Configuration shared by both providers."""

    model: str = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=300, gt=0)

    @field_validator("model")
    @classmethod
    def model_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model no puede estar vacío")
        return value


class ModelResponse(BaseModel):
    """A normalized result returned by a provider."""

    content: str = ""
    model: str
    provider: str
    error: str | None = None
