"""Unit tests for Phase 3: Multi-Tier Sentiment & Structured Culture Enrichment."""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.enrichment.gemini_enricher import GeminiEnricher
from src.enrichment.pipeline import EnrichmentPipeline
from src.enrichment.roberta_analyzer import RobertaAnalyzer
from src.enrichment.schemas import CompanyStance, RedditCultureEnrichment
from src.enrichment.vader_analyzer import VaderAnalyzer, classify_sentiment
from src.ingest.standardizer import build_canonical_record, to_canonical_dataframe


# ==============================================================================
# Tier 1: VADER Analyzer Tests
# ==============================================================================

def test_classify_sentiment_thresholds():
    assert classify_sentiment(0.5, threshold=0.20) == "positive"
    assert classify_sentiment(0.20, threshold=0.20) == "positive"
    assert classify_sentiment(-0.5, threshold=0.20) == "negative"
    assert classify_sentiment(-0.20, threshold=0.20) == "negative"
    assert classify_sentiment(0.15, threshold=0.20) == "neutral"
    assert classify_sentiment(-0.15, threshold=0.20) == "neutral"


def test_vader_analyzer_positive():
    vader = VaderAnalyzer()
    res = vader.analyze_text("I love working here! The team is amazing, supportive, and leadership is great.")
    assert res["vader_score"] >= 0.20
    assert res["vader_label"] == "positive"


def test_vader_analyzer_negative():
    vader = VaderAnalyzer()
    res = vader.analyze_text("Morale is completely ruined. Toxic managers, severe burnout, and constant micromanagement.")
    assert res["vader_score"] <= -0.20
    assert res["vader_label"] == "negative"


def test_vader_analyzer_empty():
    vader = VaderAnalyzer()
    res = vader.analyze_text("")
    assert res["vader_score"] == 0.0
    assert res["vader_label"] == "neutral"


# ==============================================================================
# Tier 2: RoBERTa Analyzer Tests
# ==============================================================================

def test_roberta_analyzer_fallback_sentiment():
    roberta = RobertaAnalyzer()
    neg_res = roberta.analyze_text("The recent layoff and PIP quotas resulted in a disaster for our team morale.")
    assert neg_res["roberta_score"] <= -0.20
    assert neg_res["roberta_label"] == "negative"

    pos_res = roberta.analyze_text("Great culture, supportive team, and wonderful work life balance.")
    assert pos_res["roberta_score"] >= 0.20
    assert pos_res["roberta_label"] == "positive"


def test_roberta_analyzer_long_text_chunking():
    """Validates that text longer than 512 tokens does not cause an index error or crash."""
    roberta = RobertaAnalyzer()
    long_text = ("Our manager announced that return to office will be strictly enforced with badge tracking. " * 50)
    res = roberta.analyze_text(long_text)
    assert "roberta_score" in res
    assert "roberta_label" in res
    assert -1.0 <= res["roberta_score"] <= 1.0


# ==============================================================================
# Tier 3: Gemini Structured Enrichment & Schema Tests
# ==============================================================================

def test_reddit_culture_enrichment_pydantic_schema():
    payload = {
        "is_culture_related": True,
        "culture_intensity": 4,
        "actor_role": "current_employee",
        "recommendation": "avoid",
        "themes": ["remote_rto_policy", "micromanagement"],
        "entity_sentiments": [
            {"company": "Amazon", "score": -0.65, "stance": "critical"}
        ],
        "negation_near_theme": False,
        "key_quote": "Morale in our building just collapsed after the 5-day mandate.",
        "sentiment_score": -0.65,
        "sentiment_label": "negative",
    }
    model = RedditCultureEnrichment.model_validate(payload)
    assert model.is_culture_related is True
    assert model.actor_role == "current_employee"
    assert len(model.themes) == 2
    assert model.entity_sentiments[0].company == "Amazon"
    assert model.entity_sentiments[0].score == -0.65


def test_gemini_enricher_mock_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    enricher = GeminiEnricher(mock_mode=True)

    text = "Andy Jassy announced 5-day RTO and our team morale in the gutter."
    res = enricher.enrich_record(
        record_id="p_test100",
        text=text,
        companies=["Amazon"],
        subreddit="amazonemployees",
        date_str="2024-09",
    )

    assert res.is_culture_related is True
    assert "remote_rto_policy" in res.themes or "leadership_management_quality" in res.themes
    assert len(res.entity_sentiments) >= 1
    assert res.entity_sentiments[0].company == "Amazon"
    assert res.sentiment_score < 0.0

    # Verify checkpoint JSONL was written and can be resumed
    checkpoint_ids = enricher.load_checkpoint_ids()
    assert "p_test100" in checkpoint_ids


# ==============================================================================
# End-to-End Enrichment Pipeline Test
# ==============================================================================

def test_enrichment_pipeline_process(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))

    recs = [
        build_canonical_record(
            post_id="p_enrich1",
            title="Amazon RTO 5-day announcement",
            body="Leadership announced mandatory 5 days in office. Team morale is tanked.",
            created_utc="2024-09-16T12:00:00Z",
            subreddit="amazonemployees",
        ),
        build_canonical_record(
            post_id="p_enrich2",
            title="Google WLB and promotions",
            body="I love the autonomy here. Great culture and supportive team.",
            created_utc="2024-09-17T14:00:00Z",
            subreddit="cscareerquestions",
        ),
    ]
    df = to_canonical_dataframe(recs)
    df["companies"] = [["Amazon"], ["Google"]]
    df["is_company_tagged"] = [True, True]

    pipeline = EnrichmentPipeline(mock_gemini=True)
    df_enriched = pipeline.process(df, save=False)

    assert len(df_enriched) == 2
    # Verify all 3 tiers are populated
    assert "vader_score" in df_enriched.columns
    assert "vader_label" in df_enriched.columns
    assert "roberta_score" in df_enriched.columns
    assert "roberta_label" in df_enriched.columns
    assert "llm_sentiment_score" in df_enriched.columns
    assert "llm_sentiment_label" in df_enriched.columns
    assert "themes" in df_enriched.columns
    assert "actor_role" in df_enriched.columns
    assert "key_quote" in df_enriched.columns

    # Verify scores are bound to [-1.0, +1.0]
    for col in ["vader_score", "roberta_score", "llm_sentiment_score"]:
        assert df_enriched[col].between(-1.0, 1.0).all()
