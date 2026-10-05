
import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from src.config import PHOENIX_COLLECTOR_ENDPOINT, PHOENIX_PROJECT_NAME

LOGGER = logging.getLogger(__name__)
_TRACER: Any | None = None
_CONFIGURED = False


def configure_observability(
    enabled: bool = True,
    *,
    provider_factory: Callable[..., Any] = TracerProvider,
    exporter_factory: Callable[..., Any] = OTLPSpanExporter,
    span_processor_factory: Callable[..., Any] = SimpleSpanProcessor,
    instrumentors: tuple[Any, ...] | None = None,
) -> bool:
    global _CONFIGURED, _TRACER

    if not enabled:
        return False
    if _CONFIGURED:
        return True

    try:
        if instrumentors is None:
            from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
            from openinference.instrumentation.langchain import LangChainInstrumentor
            from openinference.instrumentation.openai import OpenAIInstrumentor

            instrumentors = (
                LangChainInstrumentor(),
                OpenAIInstrumentor(),
                GoogleGenAIInstrumentor(),
            )

        resource = Resource.create(
            {
                "service.name": "multi-agent-api",
                "openinference.project.name": PHOENIX_PROJECT_NAME,
            }
        )
        provider = provider_factory(resource=resource)
        exporter = exporter_factory(endpoint=PHOENIX_COLLECTOR_ENDPOINT)
        provider.add_span_processor(span_processor_factory(exporter))
        tracer = provider.get_tracer("multi-agent-api")
        for instrumentor in instrumentors:
            instrumentor.instrument(tracer_provider=provider)

        _TRACER = tracer
        _CONFIGURED = True
        return True
    except Exception as error:  
        LOGGER.warning("Phoenix no se pudo inicializar: %s", error)
        return False


def _active_tracer() -> Any:
    return _TRACER or trace.get_tracer("multi-agent-api")


@contextmanager
def traced_operation(name: str, **attributes: str) -> Iterator[None]:
    with _active_tracer().start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield
