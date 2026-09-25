"""Deterministic validation node for the multi-agent workflow."""

import re

from .state import MultiAgentState, ValidationResult, get_query


def validate_results(
    query: str,
    research_result: str | None,
    analysis_result: str | None,
) -> ValidationResult:
    """Checks evidence, calculation and basic consistency before final synthesis."""
    if not research_result or research_result.startswith("No se encontró"):
        return {
            "approved": False,
            "needs_research": True,
            "needs_analysis": False,
            "reason": "Falta evidencia técnica verificable.",
        }

    if not analysis_result or analysis_result.startswith("No hay datos"):
        return {
            "approved": False,
            "needs_research": False,
            "needs_analysis": True,
            "reason": "Falta un cálculo válido a partir de la evidencia.",
        }

    if "porcentaje" in query.lower() and "%" not in analysis_result:
        return {
            "approved": False,
            "needs_research": False,
            "needs_analysis": True,
            "reason": "El análisis no contiene el porcentaje solicitado.",
        }

    evidence_numbers = re.findall(r"\d+(?:[.,]\d+)?", research_result)
    if evidence_numbers and not any(number in analysis_result for number in evidence_numbers):
        return {
            "approved": False,
            "needs_research": False,
            "needs_analysis": True,
            "reason": "El análisis no utiliza el valor numérico recuperado.",
        }

    return {
        "approved": True,
        "needs_research": False,
        "needs_analysis": False,
        "reason": "La evidencia y el análisis son suficientes para sintetizar la respuesta.",
    }


def run_validation(state: MultiAgentState) -> dict:
    """Stores the structured validation result in shared state."""
    result = validate_results(
        get_query(state),
        state.get("research_result"),
        state.get("analysis_result"),
    )
    return {
        "validation_result": result,
        "contributions": {"validation": result["reason"]},
        "step_count": state["step_count"] + 1,
        "execution_trace": ["validation"],
    }
