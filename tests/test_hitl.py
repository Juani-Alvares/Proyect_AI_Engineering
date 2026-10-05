"""HITL risk classification remains local and deterministic."""

from app.hitl import requires_human_approval, route_after_hitl


def test_hitl_detects_critical_actions_only():
    assert requires_human_approval("Eliminar registros de producción") is True
    assert requires_human_approval("Investiga el pool PostgreSQL") is False


def test_hitl_route_handles_rejection():
    assert route_after_hitl({"rejected": True}) == "rejected"
    assert route_after_hitl({"rejected": False}) == "finalize"
