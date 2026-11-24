"""State representation for the LangGraph research agent workflow."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ResearchAgentState(TypedDict):
    """The shared state passed between all nodes in the research agent graph."""

    # Initial inputs
    query: str

    # Planner outputs
    intent: str  # "quantitative" | "qualitative" | "mixed"
    target_companies: List[str]
    target_themes: List[str]
    target_subreddits: List[str]
    date_range: Optional[Dict[str, str]]
    plan_notes: str

    # SQL Tool outputs
    sql_query: Optional[str]
    sql_results: Optional[List[Dict[str, Any]]]

    # Qdrant MMR Tool outputs
    retrieved_docs: List[Dict[str, Any]]

    # Verifier & Grader outputs
    is_sample_sufficient: bool
    sample_size: int
    needs_fallback: bool
    grader_notes: str

    # Synthesizer outputs
    citations: List[Dict[str, Any]]
    response: str
    iteration_count: int
