"""Unit tests for Phase 5: Qdrant vector indexing and payload-filtered MMR retrieval."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.rag.embeddings import EmbeddingModel
from src.rag.qdrant_indexer import QdrantIndexer
from src.rag.retriever import QdrantMMRRetriever


@pytest.fixture
def test_qdrant_setup(tmp_path):
    """Sets up an isolated temporary Qdrant directory and sample indexed records."""
    db_path = tmp_path / "test_qdrant"
    embedder = EmbeddingModel()
    indexer = QdrantIndexer(storage_path=db_path, embedding_model=embedder)
    indexer.init_collection(recreate=True)

    records = [
        {
            "record_id": "p_amz_rto1",
            "post_id": "p1",
            "comment_id": "",
            "depth": 0,
            "record_type": "post",
            "subreddit": "amazonemployees",
            "author": "seattle_dev",
            "year": 2024,
            "year_month": "2024-09",
            "title": "Amazon 5-day RTO mandate discussion",
            "clean_text": "Andy Jassy announced strict 5 days a week in office requirement starting January.",
            "score": 350,
            "permalink": "https://reddit.com/r/amazonemployees/comments/p1",
            "duplicate_count": 1,
            "companies": ["Amazon"],
            "themes": ["remote_rto_policy", "leadership_management_quality"],
            "actor_role": "current_employee",
            "recommendation": "avoid",
            "culture_intensity": 4,
            "key_quote": "Andy Jassy announced strict 5 days a week in office.",
            "vader_score": -0.45,
            "roberta_score": -0.60,
            "llm_sentiment_score": -0.60,
        },
        {
            "record_id": "p_meta_eff1",
            "post_id": "p2",
            "comment_id": "",
            "depth": 0,
            "record_type": "post",
            "subreddit": "cscareerquestions",
            "author": "meta_swe",
            "year": 2023,
            "year_month": "2023-03",
            "title": "Meta Year of Efficiency and manager cuts",
            "clean_text": "Zuck announced flattening layers and cutting 10,000 more roles. Team morale is shaken.",
            "score": 520,
            "permalink": "https://reddit.com/r/cscareerquestions/comments/p2",
            "duplicate_count": 1,
            "companies": ["Meta"],
            "themes": ["micromanagement", "leadership_management_quality"],
            "actor_role": "current_employee",
            "recommendation": "mixed",
            "culture_intensity": 4,
            "key_quote": "Zuck announced flattening layers.",
            "vader_score": -0.50,
            "roberta_score": -0.55,
            "llm_sentiment_score": -0.55,
        },
        {
            "record_id": "p_google_wlb1",
            "post_id": "p3",
            "comment_id": "",
            "depth": 0,
            "record_type": "post",
            "subreddit": "cscareerquestions",
            "author": "googler",
            "year": 2024,
            "year_month": "2024-05",
            "title": "Google engineering autonomy",
            "clean_text": "Autonomy is still solid in our team and we have great work life balance.",
            "score": 210,
            "permalink": "https://reddit.com/r/cscareerquestions/comments/p3",
            "duplicate_count": 1,
            "companies": ["Google"],
            "themes": ["autonomy_trust", "work_life_balance"],
            "actor_role": "current_employee",
            "recommendation": "join",
            "culture_intensity": 2,
            "key_quote": "Autonomy is still solid in our team.",
            "vader_score": 0.60,
            "roberta_score": 0.65,
            "llm_sentiment_score": 0.65,
        },
    ]

    df = pd.DataFrame(records)
    indexer.index_records(df)
    retriever = QdrantMMRRetriever(indexer=indexer)
    return indexer, retriever


def test_embedding_model_unit_norm():
    embedder = EmbeddingModel()
    vec = embedder.embed_text("Leadership RTO policy and employee morale.")
    assert len(vec) == 768
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, 1e-4) == 1.0


def test_qdrant_retriever_company_payload_filtering(test_qdrant_setup):
    indexer, retriever = test_qdrant_setup

    # Query for Amazon with strict company filter
    results = retriever.retrieve(
        query="RTO mandate badge tracking in office",
        companies=["Amazon"],
        top_k=5,
    )

    assert len(results) >= 1
    for r in results:
        assert "Amazon" in r["companies"], "Strict company filter must prevent cross-company vector leakage"
        assert r["subreddit"] == "amazonemployees"
        assert "[r/amazonemployees, 2024-09]" in r["citation"]


def test_qdrant_retriever_actor_role_filtering(test_qdrant_setup):
    indexer, retriever = test_qdrant_setup

    # Retrieve only current employee perspectives
    results = retriever.retrieve(
        query="workplace culture",
        actor_roles=["current_employee"],
        top_k=5,
    )

    assert len(results) >= 1
    for r in results:
        assert r["role"] == "current_employee"
