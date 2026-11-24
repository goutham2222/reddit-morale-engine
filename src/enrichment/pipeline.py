"""Enrichment pipeline combining Tier 1 VADER, Tier 2 RoBERTa, and Tier 3 Gemini into enriched Parquet datasets."""

from pathlib import Path
from typing import Dict, Optional
import json
import pandas as pd

from src.common.azure_storage import StorageManager
from src.common.config import EngineConfig
from src.common.logger import get_logger
from src.enrichment.gemini_enricher import GeminiEnricher
from src.enrichment.roberta_analyzer import RobertaAnalyzer
from src.enrichment.vader_analyzer import VaderAnalyzer

logger = get_logger("enrichment_pipeline")


class EnrichmentPipeline:
    """Orchestrates multi-tier sentiment and structured cultural theme enrichment."""

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        mock_gemini: bool = False,
    ):
        self.config = config or EngineConfig.load()
        self.storage = StorageManager(self.config)
        self.vader = VaderAnalyzer()
        self.roberta = RobertaAnalyzer()
        self.gemini = GeminiEnricher(self.config, mock_mode=mock_gemini)

    def process(
        self,
        df_records: pd.DataFrame,
        sample_limit: Optional[int] = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """Runs multi-tier enrichment on an in-memory DataFrame of relevant records."""
        if len(df_records) == 0:
            return df_records

        working_df = df_records.head(sample_limit) if sample_limit else df_records.copy()
        logger.info(f"Starting Multi-Tier Enrichment Pipeline on {len(working_df):,} records...")

        # ---------------------------------------------------------------------
        # Tier 1: VADER Lexicon Sentiment
        # ---------------------------------------------------------------------
        df_vader = self.vader.enrich_dataframe(working_df, text_column="clean_text")

        # ---------------------------------------------------------------------
        # Tier 2: RoBERTa Transformer Sentiment
        # ---------------------------------------------------------------------
        df_roberta = self.roberta.enrich_dataframe(df_vader, text_column="clean_text")

        # ---------------------------------------------------------------------
        # Tier 3: Structured Gemini Culture & Theme Enrichment
        # ---------------------------------------------------------------------
        logger.info("Starting Tier 3 Gemini structured culture enrichment...")
        checkpoint_ids = self.gemini.load_checkpoint_ids()

        gemini_results = []
        for _, row in df_roberta.iterrows():
            rec_id = str(row["record_id"])
            text = str(row.get("clean_text", ""))
            comps = row.get("companies", [])
            if not isinstance(comps, list):
                comps = [comps] if pd.notnull(comps) and comps else []
            sub = str(row.get("subreddit", ""))
            date_str = str(row.get("year_month", ""))

            # Enrich (checkpoint handled inside gemini_enricher)
            enrichment = self.gemini.enrich_record(
                record_id=rec_id,
                text=text,
                companies=comps,
                subreddit=sub,
                date_str=date_str,
            )

            res_dict = enrichment.model_dump()
            res_dict["record_id"] = rec_id
            gemini_results.append(res_dict)

        df_gemini_res = pd.DataFrame(gemini_results)

        # Merge results into final enriched DataFrame
        df_final = df_roberta.copy()
        df_final["is_culture_related"] = df_gemini_res["is_culture_related"].values
        df_final["culture_intensity"] = df_gemini_res["culture_intensity"].astype("int32").values
        df_final["actor_role"] = df_gemini_res["actor_role"].values
        df_final["recommendation"] = df_gemini_res["recommendation"].values
        df_final["themes"] = df_gemini_res["themes"].values
        # Serialize complex struct list to JSON string for parquet safety
        df_final["entity_sentiments"] = [
            json.dumps(s) for s in df_gemini_res["entity_sentiments"].values
        ]
        df_final["key_quote"] = df_gemini_res["key_quote"].values
        df_final["negation_near_theme"] = df_gemini_res["negation_near_theme"].astype("bool").values
        df_final["llm_sentiment_score"] = df_gemini_res["sentiment_score"].astype("float32").values
        df_final["llm_sentiment_label"] = df_gemini_res["sentiment_label"].values

        logger.info(f"Multi-Tier Enrichment Complete for {len(df_final):,} records.")

        if save:
            self.storage.save_parquet(df_final, "enriched/enriched_records.parquet")
            logger.info("Saved enriched dataset to enriched/enriched_records.parquet.")

        return df_final

    def run(
        self,
        input_relative_path: str = "standardized/relevant_records.parquet",
        save: bool = True,
        sample_limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Loads relevant records and runs full enrichment pipeline."""
        df = self.storage.load_parquet(input_relative_path)
        return self.process(df, sample_limit=sample_limit, save=save)
