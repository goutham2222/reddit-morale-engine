"""Tagging pipeline orchestrating entity disambiguation, relevance filtering,

and two-stage deduplication into production Parquet datasets.
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from src.common.azure_storage import StorageManager
from src.common.config import EngineConfig
from src.common.logger import get_logger
from src.tagging.deduplicator import TwoStageDeduplicator
from src.tagging.entity_disambiguator import EntityDisambiguator
from src.tagging.relevance_filter import RelevanceFilter

logger = get_logger("tagging_pipeline")


class TaggingPipeline:
    """End-to-end pipeline executing Stage 2: Relevance, Disambiguation & Deduplication."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig.load()
        self.storage = StorageManager(self.config)
        self.disambiguator = EntityDisambiguator(self.config)
        self.relevance_filter = RelevanceFilter(self.config)
        self.deduplicator = TwoStageDeduplicator()

    def process(
        self,
        df_records: pd.DataFrame,
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Runs the complete tagging and deduplication pipeline on an in-memory DataFrame.

        Returns:
            Dictionary containing 'relevant_records', 'company_mentions', and 'dropped_records'.
        """
        logger.info(f"Starting Tagging Pipeline on {len(df_records):,} standardized records...")

        # Step 1: Disambiguate Company Entities
        df_tagged, df_mentions = self.disambiguator.tag_dataframe(df_records)

        # Build lookup of record_id -> tagged companies for relevance filter boosting
        tagged_lookup = dict(zip(df_tagged["record_id"], df_tagged["companies"]))

        # Step 2: Relevance Filtering (Grounding with domain lexicons and company tags)
        df_relevant, df_dropped = self.relevance_filter.filter_dataframe(
            df_tagged, tagged_records=tagged_lookup
        )

        # Step 3: Two-Stage Deduplication on Relevant Records
        df_deduped, df_clean_mentions = self.deduplicator.deduplicate(
            df_records=df_relevant, df_mentions=df_mentions
        )

        outputs = {
            "relevant_records": df_deduped,
            "company_mentions": df_clean_mentions,
            "dropped_records": df_dropped,
        }

        # Step 4: Save Parquet Datasets
        if save:
            self.storage.save_parquet(df_deduped, "standardized/relevant_records.parquet")
            if df_clean_mentions is not None:
                self.storage.save_parquet(df_clean_mentions, "standardized/company_mentions.parquet")
            self.storage.save_parquet(df_dropped, "standardized/dropped_records.parquet")
            logger.info("Saved relevant_records.parquet, company_mentions.parquet, and dropped_records.parquet.")

        return outputs

    def run(
        self,
        records_relative_path: str = "standardized/records.parquet",
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Loads canonical records from storage and executes the pipeline."""
        df_records = self.storage.load_parquet(records_relative_path)
        return self.process(df_records, save=save)
