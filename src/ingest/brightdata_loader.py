"""Loader for Bright Data Reddit posts and comments exports."""

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from src.common.logger import get_logger
from src.ingest.standardizer import build_canonical_record, to_canonical_dataframe

logger = get_logger("brightdata_loader")


class BrightDataLoader:
    """Ingests and standardizes Bright Data Reddit posts and comments files."""

    def load_posts(self, posts_path: Union[str, Path]) -> pd.DataFrame:
        """Loads and standardizes a Bright Data posts file (CSV or JSON)."""
        path = Path(posts_path)
        logger.info(f"Loading Bright Data posts from {path}")

        if path.suffix.lower() == ".json":
            df_raw = pd.read_json(path)
        else:
            df_raw = pd.read_csv(path, low_memory=False)

        records = []
        for _, row in df_raw.iterrows():
            post_id = row.get("post_id") or row.get("id") or ""
            if not post_id:
                continue

            sub = row.get("community_name") or row.get("subreddit") or ""
            author = row.get("user_posted") or row.get("author") or "[deleted]"
            date_val = row.get("date_posted") or row.get("created_utc")
            title = row.get("title") or ""
            body = row.get("description") or row.get("text") or row.get("selftext") or ""
            score = row.get("num_upvotes") or row.get("score") or row.get("post_karma") or 0
            num_comments = row.get("num_comments") or 0
            permalink = row.get("url") or row.get("permalink") or ""

            rec = build_canonical_record(
                post_id=str(post_id),
                comment_id="",
                parent_id="",
                root_comment_id="",
                depth=0,
                record_type="post",
                subreddit=str(sub),
                author=str(author),
                created_utc=date_val,
                title=str(title),
                body=str(body),
                score=score,
                num_comments=num_comments,
                permalink=str(permalink),
                source="brightdata",
            )
            records.append(rec)

        df_std = to_canonical_dataframe(records)
        logger.info(f"Standardized {len(df_std):,} Bright Data posts")
        return df_std

    def load_comments(self, comments_path: Union[str, Path]) -> pd.DataFrame:
        """Loads and standardizes a Bright Data comments file (CSV or JSON)."""
        path = Path(comments_path)
        logger.info(f"Loading Bright Data comments from {path}")

        if path.suffix.lower() == ".json":
            df_raw = pd.read_json(path)
        else:
            df_raw = pd.read_csv(path, low_memory=False)

        records = []
        for _, row in df_raw.iterrows():
            c_id = row.get("comment_id") or row.get("id") or ""
            p_id = row.get("post_id") or ""
            if not c_id or not p_id:
                continue

            par_id = row.get("parent_comment_id") or row.get("parent_id") or p_id
            root_id = row.get("root_comment_id") or c_id
            depth = row.get("depth") or 1
            sub = row.get("community_name") or row.get("subreddit") or ""
            author = row.get("user_posted") or row.get("author") or "[deleted]"
            date_val = row.get("date_posted") or row.get("created_utc")
            body = row.get("comment") or row.get("body") or row.get("text") or ""
            score = row.get("num_upvotes") or row.get("score") or 0
            permalink = row.get("url") or row.get("permalink") or ""

            rec = build_canonical_record(
                post_id=str(p_id),
                comment_id=str(c_id),
                parent_id=str(par_id),
                root_comment_id=str(root_id),
                depth=int(depth),
                record_type="comment",
                subreddit=str(sub),
                author=str(author),
                created_utc=date_val,
                title="",  # strictly empty for comments
                body=str(body),
                score=score,
                num_comments=0,
                permalink=str(permalink),
                source="brightdata",
            )
            records.append(rec)

        df_std = to_canonical_dataframe(records)
        logger.info(f"Standardized {len(df_std):,} Bright Data comments")
        return df_std

    def load_and_merge(
        self, posts_path: Union[str, Path], comments_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """Loads posts and optionally comments, merges them, and ensures deduplication."""
        df_posts = self.load_posts(posts_path)
        if comments_path and Path(comments_path).exists():
            df_comments = self.load_comments(comments_path)
            # Fill missing subreddits on comments from parent posts
            if "subreddit" in df_posts.columns and "subreddit" in df_comments.columns:
                sub_lookup = df_posts.set_index("post_id")["subreddit"].to_dict()
                empty_sub_mask = (df_comments["subreddit"] == "") | df_comments["subreddit"].isnull()
                df_comments.loc[empty_sub_mask, "subreddit"] = df_comments.loc[empty_sub_mask, "post_id"].map(
                    sub_lookup
                ).fillna("")

            merged = pd.concat([df_posts, df_comments], ignore_index=True)
        else:
            merged = df_posts

        # Drop exact primary key duplicates
        deduped = merged.drop_duplicates(subset=["record_id"]).copy()
        logger.info(f"Final standardized Bright Data dataset: {len(deduped):,} records")
        return deduped
