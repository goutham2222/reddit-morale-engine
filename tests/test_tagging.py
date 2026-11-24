"""Unit tests for workplace relevance filtering, entity disambiguation, and two-stage deduplication."""

import pandas as pd
import pytest

from src.ingest.standardizer import build_canonical_record, to_canonical_dataframe
from src.tagging.deduplicator import TwoStageDeduplicator
from src.tagging.entity_disambiguator import EntityDisambiguator
from src.tagging.pipeline import TaggingPipeline
from src.tagging.relevance_filter import RelevanceFilter


# ==============================================================================
# RelevanceFilter Tests
# ==============================================================================

def test_relevance_filter_deleted_and_empty():
    filter_mod = RelevanceFilter()
    res1 = filter_mod.evaluate_text("[deleted]", subreddit="cscareerquestions")
    assert res1["is_relevant"] is False
    assert res1["drop_reason"] == "deleted_or_empty"

    res2 = filter_mod.evaluate_text("[removed]", subreddit="jobs")
    assert res2["is_relevant"] is False

    res3 = filter_mod.evaluate_text("   ", subreddit="antiwork")
    assert res3["is_relevant"] is False


def test_relevance_filter_workplace_career_sub():
    filter_mod = RelevanceFilter()
    text = "Our manager announced new PIP quotas and work life balance has completely deteriorated."
    res = filter_mod.evaluate_text(text, subreddit="cscareerquestions")
    assert res["is_relevant"] is True
    assert res["relevance_score"] >= 0.5
    assert "manager" in res["matched_terms"] or "wlb" in res["matched_terms"] or "pip" in res["matched_terms"]


def test_relevance_filter_consumer_noise_dropped():
    filter_mod = RelevanceFilter()
    text = "Use this coupon code for free shipping on your order! Tracking number arrived late."
    res = filter_mod.evaluate_text(text, subreddit="technology")
    assert res["is_relevant"] is False
    assert "consumer_noise" in res["drop_reason"]


def test_relevance_filter_company_internal_subreddit():
    filter_mod = RelevanceFilter()
    text = "Shift lead talked about scheduling changes tomorrow."
    res = filter_mod.evaluate_text(text, subreddit="amazonemployees")
    assert res["is_relevant"] is True
    assert res["relevance_score"] == 1.0


# ==============================================================================
# EntityDisambiguator Tests
# ==============================================================================

def test_disambiguator_suppresses_monthly_meta_thread():
    """Validates that 'Monthly Meta-Thread' does NOT trigger a false positive Meta tag."""
    disambiguator = EntityDisambiguator()
    comps, mentions = disambiguator.disambiguate_text(
        record_id="p1",
        title="Monthly Meta-Thread for Career Advice",
        body="Ask any general career questions here.",
        subreddit="cscareerquestions",
    )
    assert "Meta" not in comps, "Monthly Meta-Thread should be suppressed by negative lookbehind"
    assert len(mentions) == 0


def test_disambiguator_tags_real_meta_leadership():
    """Validates that real Meta corporate announcements are accurately tagged."""
    disambiguator = EntityDisambiguator()
    comps, mentions = disambiguator.disambiguate_text(
        record_id="p2",
        title="Mark Zuckerberg announced Meta is cutting middle management",
        body="Zuck said this is the Year of Efficiency to improve culture and execution.",
        subreddit="cscareerquestions",
    )
    assert "Meta" in comps
    assert any(m["company"] == "Meta" for m in mentions)


def test_disambiguator_suppresses_apple_pie_fruit():
    """Validates that non-corporate Apple references (fruit/recipes) in general subs are suppressed."""
    disambiguator = EntityDisambiguator()
    comps, mentions = disambiguator.disambiguate_text(
        record_id="p3",
        title="Best apple pie recipe",
        body="I baked an apple pie with fresh apples.",
        subreddit="technology",
        has_workplace_keywords=False,
    )
    assert "Apple" not in comps


def test_disambiguator_tags_apple_workplace_policy():
    """Validates that Apple workplace culture posts are accurately tagged."""
    disambiguator = EntityDisambiguator()
    comps, mentions = disambiguator.disambiguate_text(
        record_id="p4",
        title="Apple return to office policy strict enforcement",
        body="Tim Cook sent a memo to staff regarding 3 days a week in Cupertino.",
        subreddit="cscareerquestions",
        has_workplace_keywords=True,
    )
    assert "Apple" in comps
    assert any(m["company"] == "Apple" for m in mentions)


def test_disambiguator_subreddit_prior_amazonemployees():
    """Validates that posts in r/amazonemployees are automatically tagged Amazon."""
    disambiguator = EntityDisambiguator()
    comps, mentions = disambiguator.disambiguate_text(
        record_id="p5",
        title="Badge swipe monitoring",
        body="Are managers tracking badge swipes now in your building?",
        subreddit="amazonemployees",
    )
    assert "Amazon" in comps
    assert any(m["match_source"] == "subreddit" for m in mentions)


def test_disambiguator_multi_company_comparison():
    """Validates that posts comparing multiple firms produce normalized bridge table mentions."""
    disambiguator = EntityDisambiguator()
    comps, mentions = disambiguator.disambiguate_text(
        record_id="p6",
        title="Comparing offer at Google vs Meta",
        body="Google offers higher base salary, but Meta offers higher equity RSUs. How is WLB?",
        subreddit="cscareerquestions",
    )
    assert "Google" in comps
    assert "Meta" in comps
    assert len(mentions) == 2
    comp_names = [m["company"] for m in mentions]
    assert "Google" in comp_names and "Meta" in comp_names


# ==============================================================================
# TwoStageDeduplicator Tests
# ==============================================================================

def test_two_stage_deduplication_viral_repost():
    """Validates that identical viral reposts are collapsed into 1 record with incremented count."""
    recs = [
        build_canonical_record(
            post_id="p_orig",
            title="VP Resignation Announcement",
            body="I am resigning in protest of corporate treatment of warehouse workers.",
            created_utc="2020-05-04T10:00:00Z",
            score=5000,
            subreddit="antiwork",
        ),
        build_canonical_record(
            post_id="p_copy1",
            title="VP Resignation Announcement",
            body="I am resigning in protest of corporate treatment of warehouse workers.",
            created_utc="2020-05-04T11:00:00Z",
            score=120,
            subreddit="cscareerquestions",
        ),
        build_canonical_record(
            post_id="p_copy2",
            title="VP Resignation Announcement",
            body="I am resigning in protest of corporate treatment of warehouse workers.",
            created_utc="2020-05-04T12:00:00Z",
            score=45,
            subreddit="jobs",
        ),
    ]
    df = to_canonical_dataframe(recs)
    df["companies"] = [["Amazon"], ["Amazon"], ["Amazon"]]
    df["is_company_tagged"] = [True, True, True]

    # Create dummy bridge table
    df_mentions = pd.DataFrame([
        {"mention_id": "m1", "record_id": "p_p_orig", "company": "Amazon", "confidence": 0.95},
        {"mention_id": "m2", "record_id": "p_p_copy1", "company": "Amazon", "confidence": 0.95},
        {"mention_id": "m3", "record_id": "p_p_copy2", "company": "Amazon", "confidence": 0.95},
    ])

    deduplicator = TwoStageDeduplicator()
    df_deduped, df_clean_mentions = deduplicator.deduplicate(df, df_mentions)

    assert len(df_deduped) == 1, "3 identical reposts must collapse into exactly 1 record"
    assert df_deduped.iloc[0]["duplicate_count"] == 3
    assert df_deduped.iloc[0]["score"] == 5000  # Kept the highest karma original
    assert len(df_clean_mentions) == 1
    assert df_clean_mentions.iloc[0]["record_id"] == df_deduped.iloc[0]["record_id"]


# ==============================================================================
# TaggingPipeline End-to-End Test
# ==============================================================================

def test_tagging_pipeline_process():
    recs = [
        build_canonical_record(
            post_id="p10",
            title="Andy Jassy 5-day RTO mandate announced",
            body="Amazon employees must be in office 5 days a week starting January.",
            created_utc="2024-09-16T14:00:00Z",
            subreddit="cscareerquestions",
        ),
        build_canonical_record(
            post_id="p11",
            title="[deleted]",
            body="[deleted]",
            created_utc="2024-09-16T15:00:00Z",
            subreddit="cscareerquestions",
        ),
    ]
    df = to_canonical_dataframe(recs)

    pipeline = TaggingPipeline()
    outputs = pipeline.process(df, save=False)

    relevant = outputs["relevant_records"]
    dropped = outputs["dropped_records"]
    mentions = outputs["company_mentions"]

    assert len(relevant) == 1
    assert relevant.iloc[0]["post_id"] == "p10"
    assert "Amazon" in relevant.iloc[0]["companies"]
    assert len(dropped) == 1
    assert dropped.iloc[0]["drop_reason"] == "deleted_or_empty"
    assert len(mentions) >= 1
    assert "Amazon" in mentions["company"].values
