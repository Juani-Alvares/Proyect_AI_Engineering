"""Research specialist with access only to the local search tool."""

from typing import Any

from langgraph.prebuilt import create_react_agent

from ..state import MultiAgentState, get_query
from ..tools import search_technical_docs

RESEARCH_PROMPT = (
    "Eres un investigador técnico. Busca evidencia concreta en documentos locales, "
    "cita el archivo de origen y no realices cálculos."
)


def create_research_react_agent(model: Any):
    """Builds the optional LangGraph ReAct specialist for an LLM-enabled run."""
    return create_react_agent(model=model, tools=[search_technical_docs], prompt=RESEARCH_PROMPT)


def run_research_agent(state: MultiAgentState) -> dict:
    """Runs deterministic local research with only the query as context."""
    query = get_query(state)
    result = search_technical_docs.invoke({"query": query})
    return {
        "research_result": result,
        "validation_result": None,
        "contributions": {"research": result},
        "step_count": state["step_count"] + 1,
        "execution_trace": ["research"],
    }
