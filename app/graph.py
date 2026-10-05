"""Production graph: the approved multi-agent graph plus an HITL checkpoint."""

from langgraph.graph import END, START, StateGraph

from src.multi_agent.agents.analyst_agent import run_analyst_agent
from src.multi_agent.agents.research_agent import run_research_agent
from src.multi_agent.graph import finalize
from src.multi_agent.state import MultiAgentState, create_initial_state
from src.multi_agent.supervisor import route_supervisor, run_supervisor
from src.multi_agent.validation import run_validation

from .hitl import hitl_node, route_after_hitl
from .observability import traced_operation


class ProductionState(MultiAgentState):
    requires_approval: bool
    approval_decision: dict | None
    rejected: bool


def rejected_node(state: ProductionState) -> dict:
    return {
        "final_answer": "La ejecución fue rechazada porque requiere aprobación humana.",
        "task_completed": True,
        "next_agent": "finish",
        "execution_trace": ["rejected"],
    }


def traced_node(name: str, handler):
    """Adds a small Phoenix/OpenTelemetry span without changing existing nodes."""
    def run(state):
        with traced_operation(name):
            return handler(state)

    return run


def create_production_graph(checkpointer=None):
    """Compiles the existing specialist workflow with a checkpoint-capable HITL node."""
    builder = StateGraph(ProductionState)
    builder.add_node("supervisor", traced_node("supervisor", run_supervisor))
    builder.add_node("research", traced_node("research", run_research_agent))
    builder.add_node("analysis", traced_node("analysis", run_analyst_agent))
    builder.add_node("validation", traced_node("validation", run_validation))
    builder.add_node("hitl", traced_node("hitl", hitl_node))
    builder.add_node("finalize", traced_node("finalize", finalize))
    builder.add_node("rejected", traced_node("rejected", rejected_node))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {"research": "research", "analysis": "analysis", "validation": "validation", "finish": "hitl"},
    )
    builder.add_edge("research", "supervisor")
    builder.add_edge("analysis", "supervisor")
    builder.add_edge("validation", "supervisor")
    builder.add_conditional_edges("hitl", route_after_hitl, {"finalize": "finalize", "rejected": "rejected"})
    builder.add_edge("finalize", END)
    builder.add_edge("rejected", END)
    return builder.compile(checkpointer=checkpointer)


def create_production_state(query: str) -> ProductionState:
    state = create_initial_state(query)
    state.update({"requires_approval": False, "approval_decision": None, "rejected": False})
    return state  # type: ignore[return-value]
