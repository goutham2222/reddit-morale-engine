"""Unit tests for text processing, base-36 ID cleaning, and hashing."""

import pytest
from src.common.text_utils import clean_reddit_id, compute_content_hash, normalize_text


def test_clean_reddit_id_standard():
    assert clean_reddit_id("t3_1oxu8rh") == "1oxu8rh"
    assert clean_reddit_id("t1_abcdef") == "abcdef"
    assert clean_reddit_id("1oxu8rh") == "1oxu8rh"


def test_clean_reddit_id_prevents_destructive_stripping():
    """Validates that base-36 Reddit IDs starting with 't' or '3' do NOT have

    their valid characters stripped.

    Legacy defect: 't3_3abcde'.strip('t3_') resulted in 'abcde' (corrupting the ID).
    """
    assert clean_reddit_id("t3_3abcde") == "3abcde"
    assert clean_reddit_id("t1_t12345") == "t12345"
    assert clean_reddit_id("t3_333333") == "333333"
    assert clean_reddit_id("3abcde") == "3abcde"
    assert clean_reddit_id("t12345") == "t12345"


def test_clean_reddit_id_null_and_whitespace():
    assert clean_reddit_id(None) == ""
    assert clean_reddit_id("") == ""
    assert clean_reddit_id("   ") == ""
    assert clean_reddit_id("nan") == ""
    assert clean_reddit_id("None") == ""


def test_normalize_text():
    raw = "Here is &amp; &lt;text&gt; with   extra   spaces\nand a newline."
    norm = normalize_text(raw)
    assert "&" in norm
    assert "<text>" in norm
    assert "\n" in norm
    assert "   " not in norm


def test_compute_content_hash_deduplication():
    text1 = "Andy Jassy announces strict 5-day RTO mandate."
    text2 = "andy jassy announces strict 5day rto mandate!"
    text3 = "Different text entirely."

    hash1 = compute_content_hash(text1)
    hash2 = compute_content_hash(text2)
    hash3 = compute_content_hash(text3)

    assert hash1 == hash2, "Normalized content hash should match regardless of punctuation/casing"
    assert hash1 != hash3, "Different content must produce different hashes"
