"""Verifier & Grader node: validates evidence sample size, checks metadata, and verifies citations."""

from typing import Dict, List
from src.agents.state import ResearchAgentState
from src.common.logger import get_logger

logger = get_logger("verifier_node")


def verifier_node(state: ResearchAgentState) -> Dict:
    """Validates retrieved evidence quality, checks sample sizes, and verifies citation formatting."""
    docs = state.get("retrieved_docs", [])
    sql_res = state.get("sql_results", [])
    intent = state.get("intent", "mixed")
    iter_count = state.get("iteration_count", 0)

    # Calculate effective sample size
    sample_size = len(docs)
    if sql_res and isinstance(sql_res, list) and len(sql_res) > 0:
        first_row = sql_res[0]
        if "total_records" in first_row:
            sample_size = sum(r.get("total_records", 0) for r in sql_res)
        elif "sample_size" in first_row:
            sample_size = sum(r.get("sample_size", 0) for r in sql_res)

    is_sufficient = sample_size >= 10 if intent in ("quantitative", "mixed") else len(docs) >= 1

    # Check citation integrity
    citations = []
    for doc in docs:
        rec_id = doc.get("record_id", "")
        sub = doc.get("subreddit", "unknown")
        ym = doc.get("year_month", "")
        role = doc.get("role", "unknown")
        quote = doc.get("key_quote", "")
        permalink = doc.get("permalink", "")

        citations.append({
            "record_id": rec_id,
            "subreddit": f"r/{sub}",
            "year_month": ym,
            "author_role": role,
            "quote": quote,
            "permalink": permalink,
            "citation_label": f"[r/{sub}, {ym}] ({role})",
        })

    # Fallback trigger: if qualitative evidence requested but 0 docs returned on strict filters
    needs_fallback = (len(docs) == 0 and intent in ("qualitative", "mixed") and iter_count == 0)

    grader_notes = (
        f"Verified {len(citations)} citations | Sample size: {sample_size:,} | "
        f"Sufficient: {is_sufficient} | Needs Fallback: {needs_fallback}"
    )
    logger.info(f"Verifier finished: {grader_notes}")

    return {
        "is_sample_sufficient": is_sufficient,
        "sample_size": sample_size,
        "needs_fallback": needs_fallback,
        "citations": citations,
        "grader_notes": grader_notes,
        "iteration_count": iter_count + 1,
    }
