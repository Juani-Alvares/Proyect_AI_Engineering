"""API tests with an in-memory Redis double and no external services."""

from fastapi.testclient import TestClient

from app.main import create_app


class FakeRedis:
    def __init__(self):
        self.data = {}

    async def set(self, key, value):
        self.data[key] = value

    async def get(self, key):
        return self.data.get(key)

    async def ping(self):
        return True


class CompleteGraph:
    async def ainvoke(self, state, config):
        return {"final_answer": "Resultado local", "task_completed": True}


def test_tasks_api_creates_and_returns_a_job():
    app = create_app(redis_client=FakeRedis(), graph=CompleteGraph(), enable_observability=False)
    with TestClient(app) as client:
        created = client.post("/tasks", json={"query": "Analiza las conexiones PostgreSQL"})
        assert created.status_code == 202
        job_id = created.json()["job_id"]

        detail = client.get(f"/tasks/{job_id}")
        assert detail.status_code == 200
        assert detail.json()["query"] == "Analiza las conexiones PostgreSQL"

        health = client.get("/health")
        assert health.json() == {"status": "ok", "redis": "ok"}


def test_tasks_api_rejects_blank_query():
    app = create_app(redis_client=FakeRedis(), graph=CompleteGraph(), enable_observability=False)
    with TestClient(app) as client:
        response = client.post("/tasks", json={"query": "   "})
    assert response.status_code == 422
