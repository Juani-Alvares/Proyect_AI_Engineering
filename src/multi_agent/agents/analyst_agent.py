"""Analysis specialist with access only to the calculation tool."""

import re
from typing import Any

from langgraph.prebuilt import create_react_agent

from ..state import MultiAgentState, get_query
from ..tools import calculate_percentage

ANALYST_PROMPT = (
    "Eres un analista técnico. Usa únicamente la evidencia recibida y realiza "
    "cálculos verificables. No investigues documentos adicionales."
)


def create_analyst_react_agent(model: Any):
    """Builds the optional LangGraph ReAct specialist for an LLM-enabled run."""
    return create_react_agent(model=model, tools=[calculate_percentage], prompt=ANALYST_PROMPT)


def _number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def run_analyst_agent(state: MultiAgentState) -> dict:
    """Calculates the requested percentage from research evidence and the query."""
    research_result = state.get("research_result") or ""
    query = get_query(state)
    maximum = _number(r"máximo de (\d+(?:[.,]\d+)?) conexiones", research_result)
    active = _number(r"(\d+(?:[.,]\d+)?) conexiones activas", query)

    if maximum is None or active is None:
        result = "No hay datos numéricos suficientes para realizar el cálculo solicitado."
    else:
        result = calculate_percentage.invoke({"part": active, "total": maximum})

    return {
        "analysis_result": result,
        "validation_result": None,
        "contributions": {"analysis": result},
        "step_count": state["step_count"] + 1,
        "execution_trace": ["analysis"],
    }
