"""Shared state for the multi-agent graph."""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

MAX_STEPS = 12


class ValidationResult(TypedDict):
    approved: bool
    needs_research: bool
    needs_analysis: bool
    reason: str


class MultiAgentState(MessagesState):
    """State shared by the supervisor, specialists, validation and final node."""

    next_agent: str
    task_completed: bool
    step_count: int
    research_result: str | None
    analysis_result: str | None
    validation_result: ValidationResult | None
    final_answer: str | None
    contributions: Annotated[dict[str, str], operator.or_]
    execution_trace: Annotated[list[str], operator.add]


def create_initial_state(query: str) -> MultiAgentState:
    """Creates an explicit initial state without requiring an API key."""
    return {
        "messages": [HumanMessage(content=query)],
        "next_agent": "",
        "task_completed": False,
        "step_count": 0,
        "research_result": None,
        "analysis_result": None,
        "validation_result": None,
        "final_answer": None,
        "contributions": {},
        "execution_trace": [],
    }


def get_query(state: MultiAgentState) -> str:
    """Gets only the user's query instead of passing the full state to agents."""
    messages = state.get("messages", [])
    if not messages:
        return ""

    message = messages[-1]
    content = message.content if hasattr(message, "content") else message["content"]
    return str(content)
