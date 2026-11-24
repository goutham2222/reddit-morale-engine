"""Two-stage deduplication engine eliminating exact and near-duplicate spam flooding.

Stage 1: Canonical primary key deduplication on record_id.
Stage 2: Normalized content hash (sha256) deduplication, collapsing viral reposts
(e.g., the 357 Tim Bray resignation copies) into a single record with incremented duplicate_count.
"""

from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from src.common.logger import get_logger

logger = get_logger("deduplicator")


class TwoStageDeduplicator:
    """Deduplicates Reddit records to prevent spam flooding in analytical marts and vector indices."""

    def deduplicate(
        self,
        df_records: pd.DataFrame,
        df_mentions: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Executes two-stage deduplication on records and updates bridge table references.

        Returns:
            deduped_records: DataFrame containing only unique texts, with duplicate_count tracked
            updated_mentions: Cleaned company_mentions bridge table mapped to surviving record_ids
        """
        initial_count = len(df_records)
        if initial_count == 0:
            return df_records, df_mentions

        logger.info(f"Starting two-stage deduplication on {initial_count:,} records...")

        # ---------------------------------------------------------------------
        # Stage 1: Canonical Primary Key Deduplication
        # ---------------------------------------------------------------------
        stage1_df = df_records.drop_duplicates(subset=["record_id"], keep="first").copy()
        pk_dropped = initial_count - len(stage1_df)
        if pk_dropped > 0:
            logger.info(f"Stage 1 (Primary Key Dedup): Removed {pk_dropped:,} duplicate record_ids.")

        # ---------------------------------------------------------------------
        # Stage 2: Normalized Content Hash Deduplication
        # ---------------------------------------------------------------------
        # Only collapse rows that have a valid non-empty content_hash
        has_hash_mask = (stage1_df["content_hash"] != "") & stage1_df["content_hash"].notnull()
        hash_df = stage1_df[has_hash_mask].copy()
        no_hash_df = stage1_df[~has_hash_mask].copy()

        # Sort to prioritize keeping the best representative record:
        # 1) depth ascending (posts over comments), 2) score descending (highest karma), 3) created_utc ascending (earliest)
        hash_df = hash_df.sort_values(
            by=["depth", "score", "created_utc"],
            ascending=[True, False, True],
        )

        id_remap: Dict[str, str] = {}  # absorbed_id -> kept_id
        grouped_records = []

        for c_hash, group in hash_df.groupby("content_hash", sort=False):
            if len(group) == 1:
                grouped_records.append(group.iloc[0].to_dict())
                continue

            # Multi-record group (viral repost or cross-post)
            primary_rec = group.iloc[0].to_dict()
            kept_id = primary_rec["record_id"]

            # Aggregate duplicate counts
            total_duplicates = int(group["duplicate_count"].sum())
            primary_rec["duplicate_count"] = total_duplicates

            # Combine company tags if present
            all_comps = set()
            for comps in group["companies"]:
                if isinstance(comps, (list, set)):
                    all_comps.update(comps)
                elif isinstance(comps, str) and comps:
                    all_comps.add(comps)
            primary_rec["companies"] = sorted(list(all_comps))
            primary_rec["is_company_tagged"] = len(all_comps) > 0

            # Map absorbed IDs to kept ID
            for absorbed_id in group["record_id"]:
                if absorbed_id != kept_id:
                    id_remap[absorbed_id] = kept_id

            grouped_records.append(primary_rec)

        stage2_hash_df = pd.DataFrame(grouped_records)
        final_df = pd.concat([stage2_hash_df, no_hash_df], ignore_index=True)

        # Ensure correct datatypes
        final_df["duplicate_count"] = final_df["duplicate_count"].astype("int32")
        final_df["score"] = final_df["score"].astype("int32")
        final_df["depth"] = final_df["depth"].astype("int32")

        hash_collapsed = len(stage1_df) - len(final_df)
        logger.info(
            f"Stage 2 (Content Hash Dedup): Collapsed {hash_collapsed:,} identical reposts. "
            f"Final unique records: {len(final_df):,}."
        )

        # ---------------------------------------------------------------------
        # Update Bridge Table references
        # ---------------------------------------------------------------------
        updated_mentions = None
        if df_mentions is not None and len(df_mentions) > 0:
            updated_mentions = df_mentions.copy()
            if id_remap:
                updated_mentions["record_id"] = updated_mentions["record_id"].map(
                    lambda r_id: id_remap.get(r_id, r_id)
                )

            # Drop bridge mentions for any records that were discarded entirely
            valid_ids = set(final_df["record_id"].unique())
            updated_mentions = updated_mentions[updated_mentions["record_id"].isin(valid_ids)].copy()

            # Deduplicate multiple bridge rows pointing to the same (record_id, company)
            updated_mentions = updated_mentions.drop_duplicates(
                subset=["record_id", "company"], keep="first"
            ).copy()

            logger.info(f"Updated Bridge Mentions: {len(updated_mentions):,} distinct company links.")

        return final_df, updated_mentions
