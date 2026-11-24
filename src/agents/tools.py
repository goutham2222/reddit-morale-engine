"""Tools for LangGraph agent: DuckDB read-only SQL executor and Qdrant MMR semantic retriever."""

from typing import Any, Dict, List, Optional
import pandas as pd

from src.analytics.duckdb_marts import DuckDBMartsManager
from src.common.config import EngineConfig
from src.common.logger import get_logger
from src.rag.retriever import QdrantMMRRetriever

logger = get_logger("agent_tools")


class DuckDBSQLTool:
    """Safe, read-only SQL execution tool querying DuckDB analytical marts."""

    def __init__(self, marts_manager: Optional[DuckDBMartsManager] = None):
        self.marts = marts_manager or DuckDBMartsManager()

    def execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        """Executes a SELECT query and returns records as a list of dicts."""
        sql_clean = sql.strip().rstrip(";")
        # Safety check: enforce read-only SELECT or WITH statements
        if not (sql_clean.upper().startswith("SELECT") or sql_clean.upper().startswith("WITH")):
            raise ValueError("Only read-only SELECT or WITH analytical queries are permitted.")

        logger.info(f"Executing DuckDB SQL query: {sql_clean[:120]}...")
        try:
            df = self.marts.query(sql_clean)
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return [{"error": str(e)}]


class QdrantMMRTool:
    """Tool for payload-filtered Maximal Marginal Relevance semantic search in Qdrant."""

    def __init__(self, retriever: Optional[QdrantMMRRetriever] = None):
        self.retriever = retriever or QdrantMMRRetriever()

    def search(
        self,
        query: str,
        companies: Optional[List[str]] = None,
        subreddits: Optional[List[str]] = None,
        themes: Optional[List[str]] = None,
        actor_roles: Optional[List[str]] = None,
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Searches Qdrant with payload metadata filters and MMR diversity reranking."""
        logger.info(f"Executing Qdrant MMR search for query: '{query[:60]}' (companies={companies})...")
        return self.retriever.retrieve(
            query=query,
            companies=companies,
            subreddits=subreddits,
            themes=themes,
            actor_roles=actor_roles,
            top_k=top_k,
        )
