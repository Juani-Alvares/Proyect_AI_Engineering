"""API-free tests for the multi-agent orchestrator."""

import asyncio

from src.multi_agent.graph import build_multi_agent_graph
from src.multi_agent.state import MAX_STEPS, create_initial_state
from src.multi_agent.supervisor import route_supervisor, run_supervisor
from src.multi_agent.tools import calculate_percentage, search_technical_docs
from src.multi_agent.validation import validate_results

QUERY = (
    "Investiga cuál es el máximo de conexiones configurado para PostgreSQL y "
    "calcula qué porcentaje representan 15 conexiones activas respecto del máximo."
)


def test_initial_state_has_explicit_fields():
    state = create_initial_state(QUERY)

    assert state["step_count"] == 0
    assert state["task_completed"] is False
    assert state["research_result"] is None
    assert state["analysis_result"] is None
    assert state["validation_result"] is None
    assert state["contributions"] == {}
    assert state["execution_trace"] == []


def test_local_tools_search_and_calculate_without_api():
    research = search_technical_docs.invoke({"query": "máximo conexiones PostgreSQL"})
    calculation = calculate_percentage.invoke({"part": 15, "total": 20})

    assert "20 conexiones" in research
    assert "75%" in calculation
    assert "No se puede calcular" in calculate_percentage.invoke({"part": 15, "total": 0})


def test_supervisor_routes_by_result_sufficiency():
    state = create_initial_state(QUERY)
    assert route_supervisor(state) == "research"

    state["research_result"] = "Fuente: database_connections.txt\nEvidencia: máximo de 20 conexiones."
    assert route_supervisor(state) == "analysis"

    state["analysis_result"] = "15 representa 75% de 20."
    assert route_supervisor(state) == "validation"

    state["validation_result"] = {
        "approved": True,
        "needs_research": False,
        "needs_analysis": False,
        "reason": "Correcto.",
    }
    assert route_supervisor(state) == "finish"


def test_validation_requests_missing_specialist_work():
    missing_research = validate_results(QUERY, None, "15 representa 75% de 20.")
    missing_analysis = validate_results(QUERY, "Fuente: database_connections.txt", None)
    inconsistent_analysis = validate_results(
        QUERY,
        "Fuente: database_connections.txt. Máximo: 20 conexiones.",
        "15 representa 75% de 10.",
    )
    approved = validate_results(QUERY, "Fuente: database_connections.txt", "15 representa 75% de 20.")

    assert missing_research["needs_research"] is True
    assert missing_analysis["needs_analysis"] is True
    assert inconsistent_analysis["needs_analysis"] is True
    assert approved["approved"] is True


def test_supervisor_refines_when_validation_requests_a_specialist():
    state = create_initial_state(QUERY)
    state["validation_result"] = {
        "approved": False,
        "needs_research": True,
        "needs_analysis": False,
        "reason": "Falta evidencia.",
    }
    assert route_supervisor(state) == "research"

    state["validation_result"] = {
        "approved": False,
        "needs_research": False,
        "needs_analysis": True,
        "reason": "Falta cálculo.",
    }
    assert route_supervisor(state) == "analysis"


def test_supervisor_stops_when_max_steps_is_reached():
    state = create_initial_state(QUERY)
    state["step_count"] = MAX_STEPS

    assert route_supervisor(state) == "finish"
    assert run_supervisor(state)["step_count"] == MAX_STEPS


def test_graph_refines_analysis_then_validates_again(monkeypatch):
    from src.multi_agent.agents.analyst_agent import run_analyst_agent

    calls = 0

    def incomplete_then_correct_analysis(state):
        nonlocal calls
        calls += 1
        if calls == 1:
            result = "No hay datos numéricos suficientes para realizar el cálculo solicitado."
            return {
                "analysis_result": result,
                "validation_result": None,
                "contributions": {"analysis": result},
                "step_count": state["step_count"] + 1,
                "execution_trace": ["analysis"],
            }
        return run_analyst_agent(state)

    monkeypatch.setattr("src.multi_agent.graph.run_analyst_agent", incomplete_then_correct_analysis)
    result = asyncio.run(build_multi_agent_graph().ainvoke(create_initial_state(QUERY)))
    trace = result["execution_trace"]

    assert trace.count("analysis") >= 2
    assert trace.count("validation") >= 2
    assert result["validation_result"]["approved"] is True
    assert result["task_completed"] is True
    assert result["step_count"] <= MAX_STEPS


def test_graph_runs_both_specialists_validation_and_finalization():
    result = asyncio.run(build_multi_agent_graph().ainvoke(create_initial_state(QUERY)))
    trace = result["execution_trace"]

    assert result["research_result"] is not None
    assert result["analysis_result"] is not None
    assert result["validation_result"] is not None
    assert result["final_answer"] is not None
    assert result["task_completed"] is True
    assert result["step_count"] <= MAX_STEPS
    assert result["validation_result"]["approved"] is True
    assert all(node in trace for node in ["supervisor", "research", "analysis", "validation", "finalize"])
    assert trace.index("research") < trace.index("analysis") < trace.index("validation") < trace.index("finalize")
