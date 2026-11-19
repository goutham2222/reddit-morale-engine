"""Data validation contracts ensuring schema integrity, non-null IDs, valid timestamps,

and zero title-to-comment text contamination.
"""

from typing import List, Tuple
import pandas as pd

from src.common.logger import get_logger

logger = get_logger("validator")

REQUIRED_COLUMNS = [
    "record_id",
    "post_id",
    "comment_id",
    "parent_id",
    "root_comment_id",
    "depth",
    "record_type",
    "subreddit",
    "author",
    "created_utc",
    "created_date",
    "year",
    "year_month",
    "title",
    "body",
    "clean_text",
    "score",
    "num_comments",
    "permalink",
    "source",
    "content_hash",
    "duplicate_count",
]


def validate_canonical_records(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validates that a DataFrame adheres to the canonical record specification.

    Returns (is_valid, error_list).
    """
    errors = []

    # 1. Check all required columns exist
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors

    if len(df) == 0:
        return True, []

    # 2. Check primary key non-null and uniqueness
    if df["record_id"].isnull().any():
        errors.append("Null values found in primary key 'record_id'")
    if df["record_id"].duplicated().any():
        dup_count = df["record_id"].duplicated().sum()
        errors.append(f"Duplicate values found in primary key 'record_id': {dup_count} duplicates")

    # 3. Check post_id non-null and no 't3_' prefixes remaining
    if df["post_id"].isnull().any() or (df["post_id"] == "").any():
        errors.append("Null or empty values found in 'post_id'")
    t3_leaks = df["post_id"].str.startswith("t3_").sum()
    if t3_leaks > 0:
        errors.append(f"Uncleaned 't3_' prefixes found in post_id: {t3_leaks} occurrences")

    # 4. Check comment_id for comments (depth >= 1)
    comment_mask = df["depth"] >= 1
    if comment_mask.any():
        null_comment_ids = df[comment_mask]["comment_id"].isnull() | (df[comment_mask]["comment_id"] == "")
        if null_comment_ids.any():
            errors.append(f"Comments (depth>=1) with null or empty comment_id: {null_comment_ids.sum()}")
        t1_leaks = df[comment_mask]["comment_id"].str.startswith("t1_").sum()
        if t1_leaks > 0:
            errors.append(f"Uncleaned 't1_' prefixes found in comment_id: {t1_leaks} occurrences")

    # 5. Check atomic text separation: Comments (depth >= 1) must have EMPTY title
    if comment_mask.any():
        title_leaks = (df[comment_mask]["title"] != "") & df[comment_mask]["title"].notnull()
        if title_leaks.any():
            errors.append(f"Atomic separation violation: {title_leaks.sum()} comments have non-empty post titles")

    # 6. Check created_utc is valid datetime
    if not pd.api.types.is_datetime64_any_dtype(df["created_utc"]):
        errors.append("'created_utc' must be a datetime64 dtype")

    # 7. Check year bounds (supporting 2015 - 2026)
    invalid_years = df[~df["year"].between(2015, 2026)]["year"].unique()
    if len(invalid_years) > 0:
        errors.append(f"Records found with year outside 2015-2026 range: {invalid_years}")

    is_valid = len(errors) == 0
    if not is_valid:
        for err in errors:
            logger.error(f"Validation failure: {err}")
    else:
        logger.info(f"DataFrame passed validation: {len(df):,} valid records")

    return is_valid, errors
