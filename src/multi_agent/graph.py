"""LangGraph topology for the multi-agent orchestrator."""

from langgraph.graph import END, START, StateGraph

from .agents.analyst_agent import run_analyst_agent
from .agents.research_agent import run_research_agent
from .state import MAX_STEPS, MultiAgentState
from .supervisor import route_supervisor, run_supervisor
from .validation import run_validation


def finalize(state: MultiAgentState) -> dict:
    """Creates a concise, validated answer instead of raw result concatenation."""
    validation = state.get("validation_result")
    if state["step_count"] >= MAX_STEPS and not (validation and validation["approved"]):
        answer = "No fue posible completar la tarea dentro del límite seguro de pasos."
    else:
        research = state.get("research_result") or "Sin evidencia disponible."
        analysis = state.get("analysis_result") or "Sin análisis disponible."
        answer = (
            f"La evidencia recuperada indica: {research} "
            f"Con esos datos, el análisis concluye: {analysis}"
        )

    step_count = min(state["step_count"] + 1, MAX_STEPS)
    return {
        "final_answer": answer,
        "task_completed": True,
        "next_agent": "finish",
        "step_count": step_count,
        "execution_trace": ["finalize"],
    }


def build_multi_agent_graph():
    """Builds the graph with a real validation/refinement loop."""
    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor", run_supervisor)
    graph.add_node("research", run_research_agent)
    graph.add_node("analysis", run_analyst_agent)
    graph.add_node("validation", run_validation)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "research": "research",
            "analysis": "analysis",
            "validation": "validation",
            "finish": "finalize",
        },
    )
    graph.add_edge("research", "supervisor")
    graph.add_edge("analysis", "supervisor")
    graph.add_edge("validation", "supervisor")
    graph.add_edge("finalize", END)
    return graph.compile()
