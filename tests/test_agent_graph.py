"""Unit tests for LangGraph multi-agent orchestration, state routing, and synthesis."""

import pytest

from src.agents.graph import create_research_graph, run_research_query
from src.agents.nodes.planner import planner_node
from src.agents.nodes.synthesizer import synthesizer_node
from src.agents.nodes.verifier import verifier_node
from src.agents.state import ResearchAgentState


def test_planner_node_target_extraction():
    state: ResearchAgentState = {
        "query": "Based on October 2024 posts, how did Amazon RTO 5-day mandate impact employee trust?",
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

    out = planner_node(state)
    assert "Amazon" in out["target_companies"]
    assert "remote_rto_policy" in out["target_themes"]
    assert out["date_range"] == {"start": "2024-10", "end": "2024-10"}


def test_verifier_node_citations_and_sample_size():
    state: ResearchAgentState = {
        "query": "Test query",
        "intent": "mixed",
        "target_companies": ["Amazon"],
        "target_themes": [],
        "target_subreddits": [],
        "date_range": None,
        "plan_notes": "",
        "sql_query": None,
        "sql_results": [{"total_records": 45}],
        "retrieved_docs": [
            {
                "record_id": "p_test1",
                "subreddit": "amazonemployees",
                "year_month": "2024-09",
                "role": "current_employee",
                "key_quote": "Morale completely dropped.",
                "permalink": "https://reddit.com/r/amazonemployees/1",
            }
        ],
        "is_sample_sufficient": False,
        "sample_size": 0,
        "needs_fallback": False,
        "grader_notes": "",
        "citations": [],
        "response": "",
        "iteration_count": 0,
    }

    out = verifier_node(state)
    assert out["is_sample_sufficient"] is True
    assert out["sample_size"] == 45
    assert len(out["citations"]) == 1
    assert out["citations"][0]["citation_label"] == "[r/amazonemployees, 2024-09] (current_employee)"


def test_synthesizer_node_markdown_formatting():
    state: ResearchAgentState = {
        "query": "How did Andy Jassy's ultimatum affect Amazon employee morale?",
        "intent": "mixed",
        "target_companies": ["Amazon"],
        "target_themes": ["remote_rto_policy"],
        "target_subreddits": [],
        "date_range": None,
        "plan_notes": "",
        "sql_query": "SELECT * FROM mart",
        "sql_results": [
            {
                "delta_roberta": -0.42,
                "pre_avg_roberta": 0.20,
                "post_avg_roberta": -0.22,
                "volume_change_pct": 85.0,
            }
        ],
        "retrieved_docs": [],
        "is_sample_sufficient": True,
        "sample_size": 120,
        "needs_fallback": False,
        "grader_notes": "",
        "citations": [
            {
                "citation_label": "[r/amazonemployees, 2023-08] (current_employee)",
                "quote": "If you can't disagree and commit it is not going to work out.",
            }
        ],
        "response": "",
        "iteration_count": 1,
    }

    out = synthesizer_node(state)
    report = out["response"]

    assert "Quantitative Findings" in report
    assert "-0.42" in report
    assert "Qualitative Themes" in report
    assert "Cited Reddit Evidence" in report
    assert "Empirical Scope & Methodological Limitations" in report


def test_compiled_research_graph_execution():
    """Validates that the compiled LangGraph workflow executes end-to-end without runtime errors."""
    query = "How did Meta Year of Efficiency impact leadership trust and middle management?"
    result = run_research_query(query)

    assert "response" in result
    assert len(result["response"]) > 100
    assert "Meta" in result.get("target_companies", [])
    assert result.get("is_sample_sufficient") is not None
