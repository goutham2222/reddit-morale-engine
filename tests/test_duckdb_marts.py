"""Unit tests for Phase 4: DuckDB analytical SQL marts, event studies, and DiD modeling."""

import pandas as pd
import pytest

from src.analytics.did_analysis import DiDAnalyzer
from src.analytics.duckdb_marts import DuckDBMartsManager
from src.analytics.event_study import EventStudyAnalyzer
from src.common.config import EngineConfig


@pytest.fixture
def sample_analytics_data():
    """Generates synthetic enriched records and company mentions for analytics testing."""
    records = [
        # Amazon records across pre/event/post periods (Layoff & RTO context)
        {
            "record_id": "p_amz_pre1",
            "post_id": "amz1",
            "comment_id": "",
            "depth": 0,
            "record_type": "post",
            "subreddit": "amazonemployees",
            "author": "dev1",
            "created_utc": pd.to_datetime("2021-10-15T10:00:00Z"),
            "created_date": pd.to_datetime("2021-10-15").date(),
            "year": 2021,
            "year_month": "2021-10",
            "title": "Amazon culture is okay",
            "clean_text": "Work life balance is reasonable right now.",
            "score": 100,
            "duplicate_count": 1,
            "vader_score": 0.35,
            "vader_label": "positive",
            "roberta_score": 0.40,
            "roberta_label": "positive",
            "is_culture_related": True,
            "culture_intensity": 2,
            "actor_role": "current_employee",
            "recommendation": "stay",
            "themes": ["work_life_balance"],
            "key_quote": "Work life balance is reasonable.",
            "negation_near_theme": False,
            "llm_sentiment_score": 0.40,
            "llm_sentiment_label": "positive",
        },
        {
            "record_id": "p_amz_post1",
            "post_id": "amz2",
            "comment_id": "",
            "depth": 0,
            "record_type": "post",
            "subreddit": "amazonemployees",
            "author": "dev2",
            "created_utc": pd.to_datetime("2023-10-15T10:00:00Z"),
            "created_date": pd.to_datetime("2023-10-15").date(),
            "year": 2023,
            "year_month": "2023-10",
            "title": "Morale after Jassy ultimatum",
            "clean_text": "Jassy disagree and commit statement destroyed morale. Massive burnout and PIP quotas.",
            "score": 850,
            "duplicate_count": 2,
            "vader_score": -0.65,
            "vader_label": "negative",
            "roberta_score": -0.70,
            "roberta_label": "negative",
            "is_culture_related": True,
            "culture_intensity": 5,
            "actor_role": "current_employee",
            "recommendation": "avoid",
            "themes": ["leadership_management_quality", "burnout_overtime_oncall"],
            "key_quote": "Jassy statement destroyed morale.",
            "negation_near_theme": False,
            "llm_sentiment_score": -0.70,
            "llm_sentiment_label": "negative",
        },
        # Apple control record (Pre and Post)
        {
            "record_id": "p_appl_pre1",
            "post_id": "app1",
            "comment_id": "",
            "depth": 0,
            "record_type": "post",
            "subreddit": "cscareerquestions",
            "author": "dev3",
            "created_utc": pd.to_datetime("2021-10-20T10:00:00Z"),
            "created_date": pd.to_datetime("2021-10-20").date(),
            "year": 2021,
            "year_month": "2021-10",
            "title": "Apple engineering culture",
            "clean_text": "Great engineering discipline and respected managers.",
            "score": 300,
            "duplicate_count": 1,
            "vader_score": 0.50,
            "vader_label": "positive",
            "roberta_score": 0.55,
            "roberta_label": "positive",
            "is_culture_related": True,
            "culture_intensity": 3,
            "actor_role": "current_employee",
            "recommendation": "join",
            "themes": ["leadership_management_quality"],
            "key_quote": "Great engineering discipline.",
            "negation_near_theme": False,
            "llm_sentiment_score": 0.55,
            "llm_sentiment_label": "positive",
        },
        {
            "record_id": "p_appl_post1",
            "post_id": "app2",
            "comment_id": "",
            "depth": 0,
            "record_type": "post",
            "subreddit": "cscareerquestions",
            "author": "dev4",
            "created_utc": pd.to_datetime("2023-10-20T10:00:00Z"),
            "created_date": pd.to_datetime("2023-10-20").date(),
            "year": 2023,
            "year_month": "2023-10",
            "title": "Apple stability amid layoffs",
            "clean_text": "Glad Apple avoided the mass tech layoffs. High job security.",
            "score": 600,
            "duplicate_count": 1,
            "vader_score": 0.45,
            "vader_label": "positive",
            "roberta_score": 0.50,
            "roberta_label": "positive",
            "is_culture_related": True,
            "culture_intensity": 3,
            "actor_role": "current_employee",
            "recommendation": "stay",
            "themes": ["leadership_management_quality"],
            "key_quote": "Glad Apple avoided the mass tech layoffs.",
            "negation_near_theme": False,
            "llm_sentiment_score": 0.50,
            "llm_sentiment_label": "positive",
        },
    ]

    mentions = [
        {"mention_id": "m1", "record_id": "p_amz_pre1", "company": "Amazon", "alias": "amazon", "match_source": "subreddit", "confidence": 1.0, "is_primary": True},
        {"mention_id": "m2", "record_id": "p_amz_post1", "company": "Amazon", "alias": "jassy", "match_source": "title", "confidence": 0.95, "is_primary": True},
        {"mention_id": "m3", "record_id": "p_appl_pre1", "company": "Apple", "alias": "apple", "match_source": "title", "confidence": 0.95, "is_primary": True},
        {"mention_id": "m4", "record_id": "p_appl_post1", "company": "Apple", "alias": "apple", "match_source": "title", "confidence": 0.95, "is_primary": True},
    ]

    return pd.DataFrame(records), pd.DataFrame(mentions)


def test_duckdb_marts_monthly_aggregation(sample_analytics_data):
    df_enriched, df_mentions = sample_analytics_data
    mgr = DuckDBMartsManager()
    mgr.register_tables(df_enriched, df_mentions)

    df_monthly = mgr.build_monthly_mart()
    assert len(df_monthly) >= 2
    assert "avg_vader_score" in df_monthly.columns
    assert "avg_roberta_score" in df_monthly.columns
    assert "net_sentiment_roberta" in df_monthly.columns

    # Verify Amazon in 2023-10 shows negative net sentiment
    amz_oct = df_monthly[(df_monthly["company"] == "Amazon") & (df_monthly["year_month"] == "2023-10")].iloc[0]
    assert amz_oct["net_sentiment_roberta"] < 0.0
    assert amz_oct["total_reposts_represented"] == 2


def test_event_study_analyzer_windowed_delta(sample_analytics_data):
    df_enriched, df_mentions = sample_analytics_data
    mgr = DuckDBMartsManager()
    mgr.register_tables(df_enriched, df_mentions)

    analyzer = EventStudyAnalyzer(marts_manager=mgr)
    event_meta = {
        "company": "Amazon",
        "anchor_month": "2023-08",
        "window_pre_months": 3,
        "window_post_months": 3,
        "leader": "Andy Jassy",
        "title": "Disagree and commit ultimatum",
    }

    res = analyzer.analyze_event("amazon_jassy_ultimatum", event_meta)
    assert res["company"] == "Amazon"
    assert res["anchor_month"] == "2023-08"
    assert res["post_records"] >= 1
    assert "delta_roberta" in res


def test_did_analyzer_estimator_calculation(sample_analytics_data):
    df_enriched, df_mentions = sample_analytics_data
    mgr = DuckDBMartsManager()
    mgr.register_tables(df_enriched, df_mentions)

    did_analyzer = DiDAnalyzer(marts_manager=mgr)
    res = did_analyzer.calculate_did_estimator(score_col="avg_roberta")

    assert "did_post_vs_pre" in res
    assert "treated_pre" in res
    assert "treated_post" in res
    assert "control_pre" in res
    assert "control_post" in res

    # In our sample data:
    # Treated (Amazon): Pre = +0.40, Post = -0.70 -> Delta = -1.10
    # Control (Apple):  Pre = +0.55, Post = +0.50 -> Delta = -0.05
    # DiD = -1.10 - (-0.05) = -1.05 (Negative DiD confirms morale degradation in Treated)
    assert res["delta_treated"] < res["delta_control"], "Treated morale drop must be steeper than Control"
    assert res["did_post_vs_pre"] < 0.0, "DiD estimator correctly captures negative morale divergence"
