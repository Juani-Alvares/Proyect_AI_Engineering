"""Pydantic models used by the clients."""

from enum import Enum
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


class NivelDeCriticidad(str, Enum):
    """Levels accepted by the technical extraction pipeline."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class TechnicalExtraction(BaseModel):
    """Validated output produced from a technical paragraph."""

    tecnologias: list[str] = Field(min_length=1)
    nivel_de_criticidad: NivelDeCriticidad
    resumen_tecnico: str = Field(min_length=1)

    @field_validator("tecnologias")
    @classmethod
    def technologies_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned_values = [value.strip() for value in values]
        if any(not value for value in cleaned_values):
            raise ValueError("tecnologias no puede contener valores vacíos")
        return cleaned_values

    @field_validator("resumen_tecnico")
    @classmethod
    def summary_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("resumen_tecnico no puede estar vacío")
        return value


class RAGResponse(BaseModel):
    """Grounded answer returned by the local RAG pipeline."""

    respuesta: str = Field(min_length=1)
    referencias: list[str] = Field(default_factory=list)

    @field_validator("respuesta")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("respuesta no puede estar vacía")
        return value

    @field_validator("referencias")
    @classmethod
    def references_must_be_source_files(cls, values: list[str]) -> list[str]:
        cleaned_values = [value.strip() for value in values]
        if any(not value or not value.endswith((".txt", ".md")) for value in cleaned_values):
            raise ValueError("referencias debe contener nombres de archivos .txt o .md")
        return cleaned_values
