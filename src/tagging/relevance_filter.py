"""Workplace & organizational culture relevance filter.

Evaluates Reddit posts and comments against domain lexicons, subreddit priors,
and noise filters to eliminate deleted content and consumer-product chatter.
"""

from typing import Dict, List, Optional, Set, Tuple
import re
import pandas as pd

from src.common.config import EngineConfig
from src.common.logger import get_logger

logger = get_logger("relevance_filter")


class RelevanceFilter:
    """Classifies Reddit posts and comments into culture-relevant vs irrelevant."""

    DELETED_MARKERS = {"[deleted]", "[removed]", ""}

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig.load()
        lex_cfg = self.config.workplace_lexicon or {}
        sub_cfg = self.config.subreddits or {}

        # 1. Workplace keywords
        raw_work_terms = lex_cfg.get("workplace_keywords", [])
        self.workplace_terms = [t.lower() for t in raw_work_terms if t]
        # Precompile regex for fast whole-word boundary matching
        self._work_patterns: List[Tuple[str, re.Pattern]] = [
            (t, re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE))
            for t in self.workplace_terms
        ]

        # 2. Blacklist / consumer noise keywords
        raw_black_terms = lex_cfg.get("blacklist_keywords", [])
        self.blacklist_terms = [t.lower() for t in raw_black_terms if t]
        self._black_patterns: List[Tuple[str, re.Pattern]] = [
            (t, re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE))
            for t in self.blacklist_terms
        ]

        # 3. Subreddit priors
        self.subreddits_cfg = sub_cfg.get("subreddits", {})

    def evaluate_text(
        self,
        clean_text: str,
        subreddit: str,
        record_type: str = "post",
        has_company_mention: bool = False,
    ) -> Dict:
        """Evaluates a single text and subreddit context for workplace relevance."""
        text = (clean_text or "").strip()
        sub = (subreddit or "").strip().lower()

        # 1. Check for deleted / removed / empty content
        stripped_del = re.sub(r"\[(deleted|removed)\]", "", text, flags=re.IGNORECASE).strip()
        if not text or not stripped_del or text.lower() in self.DELETED_MARKERS:
            return {
                "is_relevant": False,
                "relevance_score": 0.0,
                "matched_terms": [],
                "drop_reason": "deleted_or_empty",
            }

        if len(text) < 15 and not has_company_mention:
            return {
                "is_relevant": False,
                "relevance_score": 0.0,
                "matched_terms": [],
                "drop_reason": "too_short",
            }

        # 2. Check blacklist terms (consumer complaints, coupons, tech specs)
        matched_black = [
            term for term, pattern in self._black_patterns if pattern.search(text)
        ]

        # 3. Check workplace terms
        matched_work = [
            term for term, pattern in self._work_patterns if pattern.search(text)
        ]

        # Subreddit prior lookup
        sub_info = self.subreddits_cfg.get(sub, {})
        category = sub_info.get("category", "unknown")
        prior = sub_info.get("culture_prior", 0.5)
        require_work_terms = sub_info.get("require_workplace_keywords", False)

        # Company internal subreddits (e.g. r/amazonemployees)
        if category == "company_internal":
            if matched_black and not matched_work:
                return {
                    "is_relevant": False,
                    "relevance_score": 0.2,
                    "matched_terms": [],
                    "drop_reason": "consumer_noise_in_company_sub",
                }
            return {
                "is_relevant": True,
                "relevance_score": 1.0,
                "matched_terms": matched_work,
                "drop_reason": "",
            }

        # Blacklist dominance check: if consumer terms heavily outweigh workplace terms
        if len(matched_black) > 0 and len(matched_work) == 0:
            return {
                "is_relevant": False,
                "relevance_score": 0.1,
                "matched_terms": [],
                "drop_reason": f"consumer_noise: {', '.join(matched_black[:3])}",
            }

        # Career / labor subreddits (r/cscareerquestions, r/antiwork, r/layoffs)
        if category in ("career_focused", "labor_reform", "workplace_policy"):
            # Requires at least one workplace signal or confirmed company mention
            if len(matched_work) >= 1 or has_company_mention:
                score = min(1.0, prior * 0.5 + len(matched_work) * 0.15 + (0.2 if has_company_mention else 0.0))
                return {
                    "is_relevant": True,
                    "relevance_score": round(score, 3),
                    "matched_terms": matched_work,
                    "drop_reason": "",
                }
            else:
                return {
                    "is_relevant": False,
                    "relevance_score": round(prior * 0.4, 3),
                    "matched_terms": [],
                    "drop_reason": "no_workplace_keywords_in_career_sub",
                }

        # General tech / financial subreddits (r/technology, r/investing, r/stocks)
        if require_work_terms or category in ("general_tech", "financial"):
            if len(matched_work) >= 1 and (has_company_mention or len(matched_work) >= 2):
                score = min(1.0, prior * 0.4 + len(matched_work) * 0.2 + (0.2 if has_company_mention else 0.0))
                return {
                    "is_relevant": True,
                    "relevance_score": round(score, 3),
                    "matched_terms": matched_work,
                    "drop_reason": "",
                }
            else:
                return {
                    "is_relevant": False,
                    "relevance_score": round(prior * 0.3, 3),
                    "matched_terms": matched_work,
                    "drop_reason": "general_subreddit_lacks_sufficient_workplace_signal",
                }

        # Unknown / default subreddits: require at least 1 workplace keyword
        if len(matched_work) >= 1:
            score = min(1.0, 0.4 + len(matched_work) * 0.15)
            return {
                "is_relevant": True,
                "relevance_score": round(score, 3),
                "matched_terms": matched_work,
                "drop_reason": "",
            }

        return {
            "is_relevant": False,
            "relevance_score": 0.0,
            "matched_terms": [],
            "drop_reason": "no_workplace_signal",
        }

    def filter_dataframe(
        self,
        df: pd.DataFrame,
        tagged_records: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluates all records in a DataFrame, partitioning into (relevant_df, dropped_df)."""
        logger.info(f"Filtering {len(df):,} records for workplace culture relevance...")
        tagged_records = tagged_records or {}

        results = []
        for _, row in df.iterrows():
            rec_id = row.get("record_id", "")
            has_company = rec_id in tagged_records and len(tagged_records[rec_id]) > 0
            eval_res = self.evaluate_text(
                clean_text=row.get("clean_text", ""),
                subreddit=row.get("subreddit", ""),
                record_type=row.get("record_type", "post"),
                has_company_mention=has_company,
            )
            results.append(eval_res)

        eval_df = pd.DataFrame(results)
        df_annotated = df.copy()
        df_annotated["is_relevant"] = eval_df["is_relevant"].values
        df_annotated["relevance_score"] = eval_df["relevance_score"].values
        df_annotated["matched_work_terms"] = eval_df["matched_terms"].values
        df_annotated["drop_reason"] = eval_df["drop_reason"].values

        relevant_df = df_annotated[df_annotated["is_relevant"] == True].copy()
        dropped_df = df_annotated[df_annotated["is_relevant"] == False].copy()

        logger.info(
            f"Relevance Filtering Complete: {len(relevant_df):,} relevant records "
            f"({len(relevant_df)/max(1, len(df)):.1%}), {len(dropped_df):,} dropped"
        )
        return relevant_df, dropped_df
