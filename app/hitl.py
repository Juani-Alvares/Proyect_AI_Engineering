"""Human-in-the-loop para decisiones potencialmente críticas."""

from typing import Literal

from langgraph.types import interrupt

from src.multi_agent.state import get_query

CRITICAL_TERMS = (
    "eliminar",
    "borrar",
    "pago",
    "comprar",
    "enviar",
    "externo",
    "producción",
    "produccion",
    "deploy",
    "irreversible",
    "costo",
)


def requires_human_approval(query: str) -> bool:
    """Classifies risky actions without an LLM or external service."""
    normalized = query.casefold()
    return any(term in normalized for term in CRITICAL_TERMS)


def hitl_node(state: dict) -> dict:
    """Interrupts only when the query requests a potentially risky action."""
    query = get_query(state)
    if not requires_human_approval(query):
        return {"requires_approval": False, "execution_trace": ["hitl"]}

    decision = interrupt(
        {
            "message": "La tarea contiene una acción potencialmente crítica.",
            "query": query,
            "requires_approval": True,
        }
    )
    approved = bool(isinstance(decision, dict) and decision.get("approved"))
    return {
        "requires_approval": True,
        "approval_decision": decision if isinstance(decision, dict) else {"approved": False},
        "rejected": not approved,
        "execution_trace": ["hitl"],
    }


def route_after_hitl(state: dict) -> Literal["finalize", "rejected"]:
    return "rejected" if state.get("rejected") else "finalize"
