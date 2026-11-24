"""Context-aware company entity disambiguation for ~30 target enterprise firms.

Eliminates false positives (e.g., 'Monthly Meta-Thread', fruit/device chatter for Apple)
using regex negative lookbehinds, boundary matching, and workplace context gating.
"""

from typing import Dict, List, Optional, Set, Tuple
import re
import uuid
import pandas as pd

from src.common.config import EngineConfig
from src.common.logger import get_logger

logger = get_logger("entity_disambiguator")


class EntityDisambiguator:
    """Disambiguates and tags company entities mentioned in Reddit posts and comments."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig.load()
        companies_cfg = self.config.companies.get("companies", {})
        subreddits_cfg = self.config.subreddits.get("subreddits", {})

        self.companies_data: Dict[str, Dict] = {}
        self.subreddit_to_company: Dict[str, str] = {}

        # Precompile regexes for each company
        for company_name, meta in companies_cfg.items():
            aliases = meta.get("aliases", [])
            tickers = meta.get("stock_tickers", [])
            subs = meta.get("subreddits", [])
            neg_pats = meta.get("negative_patterns", [])
            req_workplace = meta.get("workplace_required_in_general_subs", True)

            # Map company subreddits
            for s in subs:
                self.subreddit_to_company[s.lower()] = company_name

            # Compile alias patterns with word boundaries
            alias_regexes = []
            for alias in aliases:
                pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
                alias_regexes.append((alias, pattern))

            # Compile ticker patterns with case sensitivity (e.g., AAPL, AMZN, MSFT, TSLA)
            ticker_regexes = []
            for ticker in tickers:
                pattern = re.compile(rf"\b{re.escape(ticker.upper())}\b")
                ticker_regexes.append((ticker, pattern))

            # Compile negative suppression patterns
            compiled_neg = [re.compile(p, re.IGNORECASE) for p in neg_pats]

            self.companies_data[company_name] = {
                "alias_regexes": alias_regexes,
                "ticker_regexes": ticker_regexes,
                "negative_patterns": compiled_neg,
                "workplace_required": req_workplace,
            }

        # Subreddit classifications
        self.subreddits_cfg = subreddits_cfg

    def disambiguate_text(
        self,
        record_id: str,
        title: str,
        body: str,
        subreddit: str,
        has_workplace_keywords: bool = True,
    ) -> Tuple[List[str], List[Dict]]:
        """Extracts confirmed company mentions from title, body, and subreddit context.

        Returns:
            matched_companies: List of canonical company names (e.g. ['Amazon', 'Google'])
            mentions: List of bridge table mention dictionaries
        """
        sub = (subreddit or "").strip().lower()
        title_text = (title or "").strip()
        body_text = (body or "").strip()
        full_text = f"{title_text} {body_text}".strip()

        matched_companies = set()
        mentions: List[Dict] = []

        # 1. Check Subreddit Prior (Direct mapping)
        if sub in self.subreddit_to_company:
            comp = self.subreddit_to_company[sub]
            matched_companies.add(comp)
            mentions.append({
                "mention_id": str(uuid.uuid4()),
                "record_id": record_id,
                "company": comp,
                "alias": sub,
                "match_source": "subreddit",
                "confidence": 1.0,
                "is_primary": True,
            })

        # Subreddit context info
        sub_info = self.subreddits_cfg.get(sub, {})
        is_workplace_sub = sub_info.get("category") in ("career_focused", "labor_reform", "workplace_policy")

        # 2. Match each company against Title and Body
        for company_name, data in self.companies_data.items():
            neg_patterns = data["negative_patterns"]
            req_work = data["workplace_required"]

            # First check negative suppression patterns
            is_suppressed = any(neg.search(full_text) for neg in neg_patterns)
            if is_suppressed:
                continue

            # Context requirement: for general subs, ambiguous companies require workplace terms
            if req_work and not is_workplace_sub and not has_workplace_keywords:
                # Do not match weak generic mentions in non-workplace contexts
                continue

            comp_found = False
            best_source = "body"
            best_alias = ""
            confidence = 0.85

            # A. Check Title first (higher priority)
            if title_text:
                for alias, pattern in data["alias_regexes"]:
                    if pattern.search(title_text):
                        comp_found = True
                        best_source = "title"
                        best_alias = alias
                        confidence = 0.95
                        break

            # B. Check Body if not found in title
            if not comp_found and body_text:
                for alias, pattern in data["alias_regexes"]:
                    if pattern.search(body_text):
                        comp_found = True
                        best_source = "body"
                        best_alias = alias
                        confidence = 0.85
                        break

            # C. Check Stock Tickers
            if not comp_found:
                for ticker, pattern in data["ticker_regexes"]:
                    if pattern.search(full_text):
                        comp_found = True
                        best_source = "ticker"
                        best_alias = ticker.upper()
                        confidence = 0.90
                        break

            if comp_found:
                matched_companies.add(company_name)
                mentions.append({
                    "mention_id": str(uuid.uuid4()),
                    "record_id": record_id,
                    "company": company_name,
                    "alias": best_alias,
                    "match_source": best_source,
                    "confidence": round(confidence, 2),
                    "is_primary": (best_source == "title" or len(matched_companies) == 1),
                })

        sorted_companies = sorted(list(matched_companies))
        return sorted_companies, mentions

    def tag_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Tags all records in df with company entities and produces the normalized bridge table.

        Returns:
            df_tagged: Copy of df with 'companies' (list) and 'is_company_tagged' (bool)
            df_mentions: Bridge table with normalized 1-to-many company mentions
        """
        logger.info(f"Disambiguating company entities across {len(df):,} records...")

        all_company_lists = []
        all_is_tagged = []
        all_mentions = []

        for _, row in df.iterrows():
            rec_id = str(row.get("record_id", ""))
            title = str(row.get("title", ""))
            body = str(row.get("body", ""))
            sub = str(row.get("subreddit", ""))

            # Quick check for workplace keyword presence if annotated
            matched_terms = row.get("matched_work_terms", None)
            has_work = bool(matched_terms) if isinstance(matched_terms, (list, set)) else True

            comps, mentions = self.disambiguate_text(
                record_id=rec_id,
                title=title,
                body=body,
                subreddit=sub,
                has_workplace_keywords=has_work,
            )

            all_company_lists.append(comps)
            all_is_tagged.append(len(comps) > 0)
            all_mentions.extend(mentions)

        df_tagged = df.copy()
        df_tagged["companies"] = all_company_lists
        df_tagged["is_company_tagged"] = all_is_tagged

        # Build mentions bridge table DataFrame
        if all_mentions:
            df_mentions = pd.DataFrame(all_mentions)
        else:
            df_mentions = pd.DataFrame(columns=[
                "mention_id", "record_id", "company", "alias", "match_source", "confidence", "is_primary"
            ])

        tagged_count = df_tagged["is_company_tagged"].sum()
        logger.info(
            f"Entity Disambiguation Complete: {tagged_count:,} records tagged with companies "
            f"({tagged_count / max(1, len(df)):.1%}), {len(df_mentions):,} total bridge mentions."
        )

        return df_tagged, df_mentions
