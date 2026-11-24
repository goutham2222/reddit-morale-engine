"""Compiled LangGraph workflow for empirical workplace research and morale synthesis."""

from typing import Any, Dict, Literal, Optional
from langgraph.graph import END, StateGraph

from src.agents.nodes.planner import planner_node
from src.agents.nodes.qdrant_retriever import qdrant_retrieval_node
from src.agents.nodes.sql_analyst import sql_analyst_node
from src.agents.nodes.synthesizer import synthesizer_node
from src.agents.nodes.verifier import verifier_node
from src.agents.state import ResearchAgentState
from src.common.logger import get_logger

logger = get_logger("research_graph")


def route_after_planner(state: ResearchAgentState) -> Literal["sql_analyst", "qdrant_retriever"]:
    """Routes execution based on whether the research task requires quantitative marts, qualitative voices, or both."""
    intent = state.get("intent", "mixed")
    if intent == "qualitative":
        return "qdrant_retriever"
    # Both quantitative and mixed start with SQL analyst
    return "sql_analyst"


def route_after_sql(state: ResearchAgentState) -> Literal["qdrant_retriever", "verifier"]:
    """Routes from SQL analyst to qualitative vector retriever for mixed queries, or straight to verifier."""
    intent = state.get("intent", "mixed")
    if intent in ("mixed", "qualitative"):
        return "qdrant_retriever"
    return "verifier"


def route_after_verifier(state: ResearchAgentState) -> Literal["qdrant_retriever", "synthesizer"]:
    """Routes to broaden retrieval if qualitative evidence was sparse, else proceeds to synthesizer."""
    if state.get("needs_fallback", False):
        return "qdrant_retriever"
    return "synthesizer"


def create_research_graph(config: Optional[Any] = None):
    """Builds and compiles the LangGraph research agent workflow."""
    workflow = StateGraph(ResearchAgentState)

    # Add core agent nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("sql_analyst", sql_analyst_node)
    workflow.add_node("qdrant_retriever", qdrant_retrieval_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Set entrypoint
    workflow.set_entry_point("planner")

    # Add conditional routing edges
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "sql_analyst": "sql_analyst",
            "qdrant_retriever": "qdrant_retriever",
        },
    )

    workflow.add_conditional_edges(
        "sql_analyst",
        route_after_sql,
        {
            "qdrant_retriever": "qdrant_retriever",
            "verifier": "verifier",
        },
    )

    workflow.add_edge("qdrant_retriever", "verifier")

    workflow.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "qdrant_retriever": "qdrant_retriever",
            "synthesizer": "synthesizer",
        },
    )

    workflow.add_edge("synthesizer", END)

    compiled_graph = workflow.compile()
    logger.info("Compiled LangGraph research workflow.")
    return compiled_graph


def run_research_query(graph_or_query: Any, query: Optional[str] = None) -> Dict:
    """Executes the complete compiled LangGraph workflow for an input research query."""
    if query is None:
        target_query = str(graph_or_query)
        graph = create_research_graph()
    else:
        graph = graph_or_query
        target_query = str(query)

    initial_state: ResearchAgentState = {
        "query": target_query,
        "intent": "mixed",
        "target_companies": [],
        "target_themes": [],
        "target_subreddits": [],
        "date_range": None,
        "plan_notes": "",
        "sql_query": None,
        "sql_results": [],
        "retrieved_docs": [],
        "is_sample_sufficient": False,
        "sample_size": 0,
        "needs_fallback": False,
        "grader_notes": "",
        "citations": [],
        "response": "",
        "iteration_count": 0,
    }

    result = graph.invoke(initial_state)
    if "final_report" not in result:
        result["final_report"] = result.get("response", "")
    if "verification_passed" not in result:
        result["verification_passed"] = result.get("is_sample_sufficient", True)
    if "sql_findings" not in result:
        result["sql_findings"] = {"records": result.get("sql_results", [])}
    return result
