"""Dynamic routing decisions for the specialized multi-agent graph."""

from typing import Literal

from .state import MAX_STEPS, MultiAgentState

Route = Literal["research", "analysis", "validation", "finish"]


def route_supervisor(state: MultiAgentState) -> Route:
    """Chooses the next specialist from the sufficiency of shared results."""
    if state["step_count"] >= MAX_STEPS:
        return "finish"

    validation = state.get("validation_result")
    if validation:
        if validation["approved"]:
            return "finish"
        if validation["needs_research"]:
            return "research"
        if validation["needs_analysis"]:
            return "analysis"

    if not state.get("research_result"):
        return "research"
    if not state.get("analysis_result"):
        return "analysis"
    if not validation:
        return "validation"
    return "finish"


def run_supervisor(state: MultiAgentState) -> dict:
    """Records the routing choice and increments the anti-loop counter."""
    if state["step_count"] >= MAX_STEPS:
        return {
            "next_agent": "finish",
            "step_count": MAX_STEPS,
            "execution_trace": ["supervisor"],
        }

    next_step = min(state["step_count"] + 1, MAX_STEPS)
    next_agent = route_supervisor({**state, "step_count": next_step})
    return {
        "next_agent": next_agent,
        "step_count": next_step,
        "execution_trace": ["supervisor"],
    }
