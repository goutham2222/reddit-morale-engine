"""Qdrant retriever node: executes payload-filtered MMR semantic search in persistent Qdrant."""

from typing import Dict
from src.agents.state import ResearchAgentState
from src.agents.tools import QdrantMMRTool
from src.common.logger import get_logger

logger = get_logger("qdrant_retriever_node")


def qdrant_retrieval_node(state: ResearchAgentState) -> Dict:
    """Retrieves qualitative employee evidence, quotes, and themes using payload-filtered MMR."""
    query = state.get("query", "")
    companies = state.get("target_companies", [])
    themes = state.get("target_themes", [])
    subreddits = state.get("target_subreddits", [])

    tool = QdrantMMRTool()
    docs = tool.search(
        query=query,
        companies=companies if companies else None,
        themes=themes if themes else None,
        subreddits=subreddits if subreddits else None,
        top_k=8,
    )

    logger.info(f"Retrieved {len(docs)} diverse evidence items from Qdrant.")
    return {"retrieved_docs": docs}
