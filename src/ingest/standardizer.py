"""Canonical standardization logic for Reddit posts and comments."""

import numpy as np
import pandas as pd

from src.common.logger import get_logger
from src.common.text_utils import clean_reddit_id, compute_content_hash, normalize_text
from src.ingest.validator import REQUIRED_COLUMNS, validate_canonical_records

logger = get_logger("standardizer")


def build_canonical_record(
    post_id: str,
    comment_id: str = "",
    parent_id: str = "",
    root_comment_id: str = "",
    depth: int = 0,
    record_type: str = "post",
    subreddit: str = "",
    author: str = "[deleted]",
    created_utc: any = None,
    title: str = "",
    body: str = "",
    score: int = 0,
    num_comments: int = 0,
    permalink: str = "",
    source: str = "unknown",
) -> dict:
    """Builds a single dictionary adhering to the canonical schema."""
    clean_p_id = clean_reddit_id(post_id)
    clean_c_id = clean_reddit_id(comment_id)
    clean_par_id = clean_reddit_id(parent_id)
    clean_root_id = clean_reddit_id(root_comment_id)

    # Derive record_id
    if record_type == "post" or depth == 0 or not clean_c_id:
        record_id = f"p_{clean_p_id}"
        record_type = "post"
        depth = 0
        clean_c_id = ""
        # For posts, parent_id is empty
        clean_par_id = ""
        norm_title = normalize_text(title)
        norm_body = normalize_text(body)
        # clean_text for post combines title and body
        if norm_title and norm_body:
            clean_text = f"{norm_title}\n\n{norm_body}"
        else:
            clean_text = norm_title or norm_body
    else:
        record_id = f"c_{clean_c_id}"
        record_type = "comment"
        if depth == 0:
            depth = 1
        # Comments NEVER store post titles directly
        norm_title = ""
        norm_body = normalize_text(body)
        clean_text = norm_body

    # Normalize timestamp
    dt = pd.to_datetime(created_utc, utc=True, errors="coerce")
    if pd.isnull(dt):
        dt = pd.Timestamp.utcnow()

    created_date = dt.date()
    year = int(dt.year)
    year_month = f"{dt.year:04d}-{dt.month:02d}"

    # Clean subreddit
    sub = (subreddit or "").strip().lower()
    if sub.startswith("r/"):
        sub = sub[2:]

    # Content hash
    c_hash = compute_content_hash(clean_text)

    return {
        "record_id": record_id,
        "post_id": clean_p_id,
        "comment_id": clean_c_id,
        "parent_id": clean_par_id,
        "root_comment_id": clean_root_id,
        "depth": int(depth),
        "record_type": record_type,
        "subreddit": sub,
        "author": (author or "[deleted]").strip(),
        "created_utc": dt,
        "created_date": created_date,
        "year": year,
        "year_month": year_month,
        "title": norm_title,
        "body": norm_body,
        "clean_text": clean_text,
        "score": int(score) if pd.notnull(score) else 0,
        "num_comments": int(num_comments) if pd.notnull(num_comments) else 0,
        "permalink": (permalink or "").strip(),
        "source": source,
        "content_hash": c_hash,
        "duplicate_count": 1,
    }


def to_canonical_dataframe(records: list) -> pd.DataFrame:
    """Converts a list of record dictionaries into a validated, typed canonical DataFrame."""
    if not records:
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    else:
        df = pd.DataFrame(records)

    # Enforce exact column ordering
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[REQUIRED_COLUMNS].copy()

    # Cast dtypes
    df["depth"] = df["depth"].fillna(0).astype("int32")
    df["score"] = df["score"].fillna(0).astype("int32")
    df["num_comments"] = df["num_comments"].fillna(0).astype("int32")
    df["duplicate_count"] = df["duplicate_count"].fillna(1).astype("int32")
    df["year"] = df["year"].astype("int32")
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)

    for str_col in [
        "record_id",
        "post_id",
        "comment_id",
        "parent_id",
        "root_comment_id",
        "record_type",
        "subreddit",
        "author",
        "year_month",
        "title",
        "body",
        "clean_text",
        "permalink",
        "source",
        "content_hash",
    ]:
        df[str_col] = df[str_col].fillna("").astype(str)

    # Validate
    is_valid, errors = validate_canonical_records(df)
    if not is_valid:
        raise ValueError(f"Standardized DataFrame failed validation: {errors}")

    return df
