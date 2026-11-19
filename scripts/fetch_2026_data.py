"""Incremental 2026 Reddit data fetcher using PRAW.

Deduplicates against existing Parquet records to fetch only net-new submissions and comments.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Set
import pandas as pd

from src.common.azure_storage import StorageManager
from src.common.config import EngineConfig
from src.common.logger import get_logger
from src.ingest.standardizer import build_canonical_record, to_canonical_dataframe

logger = get_logger("fetch_2026_data")


def get_existing_ids(storage: StorageManager) -> Set[str]:
    """Retrieves existing post_id and comment_id sets from records.parquet to prevent duplicate pulls."""
    existing_ids = set()
    parquet_path = "standardized/records.parquet"
    if storage.exists(parquet_path):
        try:
            df_existing = storage.load_parquet(parquet_path)
            for pid in df_existing["post_id"].dropna().unique():
                if pid:
                    existing_ids.add(str(pid))
            for cid in df_existing["comment_id"].dropna().unique():
                if cid:
                    existing_ids.add(str(cid))
            logger.info(f"Loaded {len(existing_ids):,} existing Reddit IDs from {parquet_path}")
        except Exception as e:
            logger.warning(f"Could not load existing Parquet records: {e}")
    return existing_ids


def fetch_2026_incremental(
    subreddits: list,
    limit_per_sub: int = 100,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Incrementally fetches 2026 posts and comments from Reddit for the target subreddits."""
    config = EngineConfig.load()
    storage = StorageManager(config)
    existing_ids = get_existing_ids(storage)

    if not config.reddit_client_id or not config.reddit_client_secret:
        logger.warning("REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET not configured. Running in mock/dry-run mode.")
        if dry_run or not config.reddit_client_id:
            # Produce a sample 2026 record for verification
            mock_records = [
                build_canonical_record(
                    post_id="mock2026p1",
                    comment_id="",
                    depth=0,
                    record_type="post",
                    subreddit="cscareerquestions",
                    author="sample_dev_2026",
                    created_utc="2026-02-15T12:00:00Z",
                    title="2026 RTO 5-day enforcement and hiring freeze updates",
                    body="Anyone else seeing strict badge swipe monitoring this month across tech?",
                    score=142,
                    num_comments=58,
                    permalink="https://reddit.com/r/cscareerquestions/comments/mock2026p1",
                    source="praw_scraper_2026",
                )
            ]
            return to_canonical_dataframe(mock_records)

    import praw

    reddit = praw.Reddit(
        client_id=config.reddit_client_id,
        client_secret=config.reddit_client_secret,
        user_agent=config.reddit_user_agent,
    )

    new_records = []
    start_2026 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()

    for sub_name in subreddits:
        logger.info(f"Checking r/{sub_name} for 2026 submissions...")
        try:
            sub = reddit.subreddit(sub_name)
            for submission in sub.new(limit=limit_per_sub):
                if submission.created_utc < start_2026:
                    continue

                clean_pid = submission.id
                if clean_pid in existing_ids:
                    continue

                # Add post
                post_rec = build_canonical_record(
                    post_id=clean_pid,
                    comment_id="",
                    parent_id="",
                    root_comment_id="",
                    depth=0,
                    record_type="post",
                    subreddit=sub_name,
                    author=str(submission.author) if submission.author else "[deleted]",
                    created_utc=submission.created_utc,
                    title=submission.title or "",
                    body=submission.selftext or "",
                    score=submission.score,
                    num_comments=submission.num_comments,
                    permalink=f"https://reddit.com{submission.permalink}",
                    source="praw_scraper_2026",
                )
                new_records.append(post_rec)
                existing_ids.add(clean_pid)

                # Fetch top comments without deep recursion to respect rate limits
                try:
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list()[:20]:
                        cid = comment.id
                        if cid in existing_ids:
                            continue
                        com_rec = build_canonical_record(
                            post_id=clean_pid,
                            comment_id=cid,
                            parent_id=comment.parent_id,
                            root_comment_id=comment.id,
                            depth=1,
                            record_type="comment",
                            subreddit=sub_name,
                            author=str(comment.author) if comment.author else "[deleted]",
                            created_utc=comment.created_utc,
                            title="",
                            body=comment.body or "",
                            score=comment.score,
                            num_comments=0,
                            permalink=f"https://reddit.com{comment.permalink}",
                            source="praw_scraper_2026",
                        )
                        new_records.append(com_rec)
                        existing_ids.add(cid)
                except Exception as e:
                    logger.debug(f"Failed to fetch comments for {clean_pid}: {e}")

        except Exception as e:
            logger.error(f"Error querying r/{sub_name}: {e}")

    logger.info(f"Fetched {len(new_records):,} net-new 2026 records.")
    return to_canonical_dataframe(new_records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch incremental 2026 Reddit data")
    parser.add_argument("--limit", type=int, default=100, help="Max posts to inspect per subreddit")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing to disk")
    args = parser.parse_args()

    cfg = EngineConfig.load()
    target_subs = list(cfg.subreddits.get("subreddits", {}).keys()) or [
        "cscareerquestions",
        "ExperiencedDevs",
        "layoffs",
        "amazonemployees",
    ]

    df_new = fetch_2026_incremental(target_subs, limit_per_sub=args.limit, dry_run=args.dry_run)
    if not args.dry_run and len(df_new) > 0:
        store = StorageManager(cfg)
        store.save_parquet(df_new, "raw/reddit_2026_incremental.parquet")
        logger.info(f"Saved {len(df_new):,} records to raw/reddit_2026_incremental.parquet")
