"""Observability tests without Phoenix or external HTTP requests."""

from app import observability


class FakeSpan:
    def __init__(self) -> None:
        self.attributes = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def set_attribute(self, key, value) -> None:
        self.attributes[key] = value


class FakeTracer:
    def __init__(self) -> None:
        self.names = []
        self.span = FakeSpan()

    def start_as_current_span(self, name):
        self.names.append(name)
        return self.span


class FakeProvider:
    def __init__(self) -> None:
        self.tracer = FakeTracer()
        self.requested_name = None
        self.processors = []

    def get_tracer(self, name):
        self.requested_name = name
        return self.tracer

    def add_span_processor(self, processor):
        self.processors.append(processor)


class FakeExporter:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def export(self, spans):
        return None

    def shutdown(self):
        return None

    def force_flush(self, timeout_millis=30_000):
        return True


class FakeSpanProcessor:
    def __init__(self, exporter):
        self.exporter = exporter


class FakeGoogleInstrumentor:
    def __init__(self) -> None:
        self.provider = None

    def instrument(self, *, tracer_provider):
        self.provider = tracer_provider


def test_observability_is_safe_when_disabled(monkeypatch):
    monkeypatch.setattr(observability, "_TRACER", None)
    monkeypatch.setattr(observability, "_CONFIGURED", False)
    assert observability.configure_observability(enabled=False) is False
    with observability.traced_operation("test-span", task_id="local"):
        assert True


def test_configured_operations_use_the_provider_tracer(monkeypatch):
    provider = FakeProvider()
    google_instrumentor = FakeGoogleInstrumentor()
    monkeypatch.setattr(observability, "_TRACER", None)
    monkeypatch.setattr(observability, "_CONFIGURED", False)
    monkeypatch.setattr(observability, "PHOENIX_PROJECT_NAME", "multi-agent-api")
    monkeypatch.setattr(observability, "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")

    def fake_provider_factory(**kwargs):
        assert kwargs["resource"].attributes["openinference.project.name"] == "multi-agent-api"
        return provider

    assert observability.configure_observability(
        provider_factory=fake_provider_factory,
        exporter_factory=FakeExporter,
        span_processor_factory=FakeSpanProcessor,
        instrumentors=(google_instrumentor,),
    ) is True
    with observability.traced_operation("worker", job_id="job-1"):
        pass

    assert provider.requested_name == "multi-agent-api"
    assert provider.processors[0].exporter.endpoint.endswith("/v1/traces")
    assert google_instrumentor.provider is provider
    assert provider.tracer.names == ["worker"]
    assert provider.tracer.span.attributes == {"job_id": "job-1"}


def test_configuration_is_idempotent(monkeypatch):
    provider = FakeProvider()
    calls = []
    monkeypatch.setattr(observability, "_TRACER", None)
    monkeypatch.setattr(observability, "_CONFIGURED", False)

    def fake_provider_factory(**kwargs):
        calls.append(kwargs)
        return provider

    options = {
        "provider_factory": fake_provider_factory,
        "exporter_factory": FakeExporter,
        "span_processor_factory": FakeSpanProcessor,
        "instrumentors": (),
    }
    observability.configure_observability(**options)
    observability.configure_observability(**options)
    assert len(calls) == 1


def test_google_genai_instrumentor_is_available_without_requests():
    from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

    assert GoogleGenAIInstrumentor is not None


def test_public_tracer_provider_initializes_without_private_exporter_access(monkeypatch):
    monkeypatch.setattr(observability, "_TRACER", None)
    monkeypatch.setattr(observability, "_CONFIGURED", False)
    assert observability.configure_observability(
        exporter_factory=FakeExporter,
        instrumentors=(),
    ) is True
    assert observability._TRACER is not None
