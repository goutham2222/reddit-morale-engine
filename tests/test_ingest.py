"""Unit tests for canonical ingestion, standardization, and Pandera contracts."""

import pandas as pd
import pytest
from src.common.azure_storage import StorageManager
from src.ingest.standardizer import build_canonical_record, to_canonical_dataframe
from src.ingest.validator import validate_canonical_records


def test_build_canonical_record_post():
    rec = build_canonical_record(
        post_id="t3_1oxu8rh",
        comment_id="",
        depth=0,
        record_type="post",
        subreddit="r/cscareerquestions",
        author="eng_worker",
        created_utc="2026-02-01T15:30:00Z",
        title="Leadership RTO Mandate Discussion",
        body="Management just announced 5-day in office requirements.",
        score=250,
        num_comments=45,
        permalink="https://reddit.com/r/cscareerquestions/comments/1oxu8rh",
        source="test",
    )

    assert rec["record_id"] == "p_1oxu8rh"
    assert rec["post_id"] == "1oxu8rh"
    assert rec["comment_id"] == ""
    assert rec["depth"] == 0
    assert rec["record_type"] == "post"
    assert rec["subreddit"] == "cscareerquestions"
    assert rec["year"] == 2026
    assert rec["year_month"] == "2026-02"
    assert rec["title"] == "Leadership RTO Mandate Discussion"
    assert "5-day in office" in rec["clean_text"]


def test_build_canonical_record_comment_zero_title_leakage():
    rec = build_canonical_record(
        post_id="t3_1oxu8rh",
        comment_id="t1_c98765",
        parent_id="t3_1oxu8rh",
        root_comment_id="t1_c98765",
        depth=1,
        record_type="comment",
        subreddit="cscareerquestions",
        author="anonymous_coworker",
        created_utc="2026-02-01T16:00:00Z",
        title="This should not appear in comment row",
        body="Completely agree, morale is in the gutter.",
        score=15,
        source="test",
    )

    assert rec["record_id"] == "c_c98765"
    assert rec["post_id"] == "1oxu8rh"
    assert rec["comment_id"] == "c98765"
    assert rec["depth"] == 1
    assert rec["record_type"] == "comment"
    # Zero title leakage contract
    assert rec["title"] == "", "Comments must have empty title to prevent keyword count inflation"
    assert rec["clean_text"] == "Completely agree, morale is in the gutter."


def test_canonical_dataframe_validation_success():
    records = [
        build_canonical_record(
            post_id="p1",
            title="Post title",
            body="Post text",
            created_utc="2025-05-10T10:00:00Z",
            subreddit="jobs",
        ),
        build_canonical_record(
            post_id="p1",
            comment_id="c1",
            depth=1,
            record_type="comment",
            body="Comment reply",
            created_utc="2025-05-10T11:00:00Z",
            subreddit="jobs",
        ),
    ]
    df = to_canonical_dataframe(records)
    is_valid, errors = validate_canonical_records(df)
    assert is_valid is True
    assert len(errors) == 0
    assert len(df) == 2


def test_canonical_dataframe_flags_title_leakage():
    """Validates that validator flags any comment contaminated with a post title."""
    records = [
        build_canonical_record(
            post_id="p1",
            comment_id="c1",
            depth=1,
            record_type="comment",
            body="Comment reply",
            created_utc="2025-05-10T11:00:00Z",
            subreddit="jobs",
        )
    ]
    df = to_canonical_dataframe(records)
    # Manually inject title leak to test validator enforcement
    df["title"] = "Leaked post title"
    is_valid, errors = validate_canonical_records(df)
    assert is_valid is False
    assert any("Atomic separation violation" in err for err in errors)


def test_storage_manager_roundtrip_parquet(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "")

    storage = StorageManager()
    records = [
        build_canonical_record(
            post_id="p100",
            title="Test Post",
            body="Some text",
            created_utc="2024-03-15T08:00:00Z",
            subreddit="antiwork",
        )
    ]
    df_orig = to_canonical_dataframe(records)
    saved_path = storage.save_parquet(df_orig, "standardized/records.parquet")

    assert saved_path.exists()
    assert saved_path.suffix == ".parquet"

    df_loaded = storage.load_parquet("standardized/records.parquet")
    assert len(df_loaded) == 1
    assert df_loaded.iloc[0]["post_id"] == "p100"
    assert df_loaded.iloc[0]["record_id"] == "p_p100"
