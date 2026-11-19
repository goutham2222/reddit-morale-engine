"""Loader for PRAW scraper Reddit exports (CSV/JSON)."""

from pathlib import Path
from typing import Union
import pandas as pd

from src.common.logger import get_logger
from src.ingest.standardizer import build_canonical_record, to_canonical_dataframe

logger = get_logger("praw_loader")


class PRAWLoader:
    """Ingests and standardizes PRAW scraper Reddit data files."""

    def load_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Loads and standardizes a PRAW scraper export file."""
        path = Path(file_path)
        logger.info(f"Loading PRAW scraper data from {path}")

        if path.suffix.lower() == ".json":
            df_raw = pd.read_json(path)
        else:
            df_raw = pd.read_csv(path, low_memory=False)

        records = []
        for _, row in df_raw.iterrows():
            p_id = str(row.get("post_id") or row.get("id") or "").strip()
            if not p_id:
                continue

            c_id = str(row.get("comment_id") or "").strip()
            depth = row.get("depth", 0)
            if pd.isnull(depth):
                depth = 1 if c_id else 0
            else:
                depth = int(depth)

            rec_type = "comment" if (c_id or depth > 0) else "post"
            par_id = str(row.get("parent_id") or row.get("parent_comment_id") or (p_id if rec_type == "comment" else "")).strip()
            root_id = str(row.get("root_comment_id") or (c_id if rec_type == "comment" else "")).strip()

            sub = str(row.get("subreddit") or row.get("community_name") or "")
            author = str(row.get("user_posted") or row.get("author") or "[deleted]")

            date_val = row.get("date_posted") or row.get("created_utc")
            if pd.isnull(date_val):
                date_val = row.get("created_iso")

            title = str(row.get("title") or "") if rec_type == "post" else ""
            if rec_type == "post":
                body = str(row.get("text") or row.get("selftext") or row.get("body") or "")
            else:
                body = str(row.get("comment_body") or row.get("comment") or row.get("text") or row.get("body") or "")

            score = row.get("score") or row.get("num_upvotes") or row.get("post_karma") or 0
            num_comments = row.get("num_comments") or 0
            permalink = str(row.get("url") or row.get("permalink") or "")

            rec = build_canonical_record(
                post_id=p_id,
                comment_id=c_id,
                parent_id=par_id,
                root_comment_id=root_id,
                depth=depth,
                record_type=rec_type,
                subreddit=sub,
                author=author,
                created_utc=date_val,
                title=title,
                body=body,
                score=score,
                num_comments=num_comments,
                permalink=permalink,
                source="praw_scraper",
            )
            records.append(rec)

        df_std = to_canonical_dataframe(records)
        deduped = df_std.drop_duplicates(subset=["record_id"]).copy()
        logger.info(f"Standardized {len(deduped):,} PRAW records")
        return deduped
