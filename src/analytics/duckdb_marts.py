"""DuckDB analytical marts manager creating SQL views directly over Parquet datasets."""

from pathlib import Path
from typing import Dict, Optional, Union
import duckdb
import pandas as pd

from src.common.azure_storage import StorageManager
from src.common.config import EngineConfig
from src.common.logger import get_logger

logger = get_logger("duckdb_marts")


class DuckDBMartsManager:
    """Manages embedded DuckDB analytical views and aggregated Parquet data marts."""

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        db_path: Optional[Union[str, Path]] = None,
    ):
        self.config = config or EngineConfig.load()
        self.storage = StorageManager(self.config)
        self.db_path = str(db_path) if db_path else ":memory:"
        self.conn = duckdb.connect(database=self.db_path)
        logger.info(f"Initialized DuckDB connection ({self.db_path})")

    def register_tables(
        self,
        df_enriched: Optional[pd.DataFrame] = None,
        df_mentions: Optional[pd.DataFrame] = None,
    ):
        """Registers in-memory DataFrames or points DuckDB views directly to Parquet files."""
        # 1. Enriched records
        if df_enriched is not None:
            self.conn.register("enriched_records_raw", df_enriched)
        else:
            p_enriched = self.storage.local_root / "enriched/enriched_records.parquet"
            if p_enriched.exists():
                self.conn.execute(
                    f"CREATE OR REPLACE VIEW enriched_records_raw AS SELECT * FROM read_parquet('{p_enriched}')"
                )
            else:
                logger.warning(f"Enriched records Parquet not found at {p_enriched}")

        # 2. Company mentions bridge table
        if df_mentions is not None:
            self.conn.register("company_mentions_raw", df_mentions)
        else:
            p_mentions = self.storage.local_root / "standardized/company_mentions.parquet"
            if p_mentions.exists():
                self.conn.execute(
                    f"CREATE OR REPLACE VIEW company_mentions_raw AS SELECT * FROM read_parquet('{p_mentions}')"
                )
            else:
                logger.warning(f"Company mentions Parquet not found at {p_mentions}")

        # 3. Create unified company-enriched join view
        self.conn.execute("""
            CREATE OR REPLACE VIEW company_enriched_view AS
            SELECT 
                m.company,
                m.alias,
                m.match_source,
                m.confidence,
                m.is_primary,
                e.record_id,
                e.post_id,
                e.comment_id,
                e.depth,
                e.record_type,
                e.subreddit,
                e.author,
                e.created_utc,
                e.created_date,
                e.year,
                e.year_month,
                e.title,
                e.clean_text,
                e.score,
                e.duplicate_count,
                e.vader_score,
                e.vader_label,
                e.roberta_score,
                e.roberta_label,
                e.is_culture_related,
                e.culture_intensity,
                e.actor_role,
                e.recommendation,
                e.themes,
                e.key_quote,
                e.negation_near_theme,
                e.llm_sentiment_score,
                e.llm_sentiment_label
            FROM enriched_records_raw e
            JOIN company_mentions_raw m ON e.record_id = m.record_id
        """)
        logger.info("Created DuckDB view: company_enriched_view")

    def build_monthly_mart(self) -> pd.DataFrame:
        """Aggregates employee morale and sentiment metrics by Company and Year-Month."""
        sql = """
            SELECT 
                company,
                year_month,
                year,
                COUNT(*) AS total_records,
                SUM(CASE WHEN depth = 0 THEN 1 ELSE 0 END) AS post_count,
                SUM(CASE WHEN depth >= 1 THEN 1 ELSE 0 END) AS comment_count,
                COUNT(DISTINCT author) AS unique_authors,
                SUM(duplicate_count) AS total_reposts_represented,
                
                -- Mean sentiment scores across tiers
                ROUND(AVG(vader_score), 4) AS avg_vader_score,
                ROUND(AVG(roberta_score), 4) AS avg_roberta_score,
                ROUND(AVG(llm_sentiment_score), 4) AS avg_llm_score,
                
                -- Classifications distribution
                SUM(CASE WHEN roberta_label = 'positive' THEN 1 ELSE 0 END) AS roberta_pos_count,
                SUM(CASE WHEN roberta_label = 'negative' THEN 1 ELSE 0 END) AS roberta_neg_count,
                SUM(CASE WHEN roberta_label = 'neutral' THEN 1 ELSE 0 END) AS roberta_neu_count,
                
                -- Net sentiment: (Pos - Neg) / Total
                ROUND((SUM(CASE WHEN roberta_label = 'positive' THEN 1 ELSE 0 END) - 
                       SUM(CASE WHEN roberta_label = 'negative' THEN 1 ELSE 0 END)) * 1.0 / COUNT(*), 4) AS net_sentiment_roberta,
                
                ROUND(AVG(culture_intensity), 2) AS avg_culture_intensity
            FROM company_enriched_view
            GROUP BY company, year_month, year
            ORDER BY company, year_month
        """
        df_monthly = self.conn.execute(sql).fetchdf()
        logger.info(f"Generated monthly mart: {len(df_monthly):,} company-month aggregates.")
        return df_monthly

    def build_quarterly_mart(self) -> pd.DataFrame:
        """Aggregates metrics by Company, Year, and Quarter."""
        sql = """
            SELECT 
                company,
                year,
                'Q' || CAST(((CAST(SUBSTR(year_month, 6, 2) AS INTEGER) - 1) / 3 + 1) AS VARCHAR) AS quarter,
                year || 'Q' || CAST(((CAST(SUBSTR(year_month, 6, 2) AS INTEGER) - 1) / 3 + 1) AS VARCHAR) AS year_quarter,
                COUNT(*) AS total_records,
                COUNT(DISTINCT author) AS unique_authors,
                SUM(duplicate_count) AS total_reposts_represented,
                ROUND(AVG(vader_score), 4) AS avg_vader_score,
                ROUND(AVG(roberta_score), 4) AS avg_roberta_score,
                ROUND(AVG(llm_sentiment_score), 4) AS avg_llm_score,
                ROUND((SUM(CASE WHEN roberta_label = 'positive' THEN 1 ELSE 0 END) - 
                       SUM(CASE WHEN roberta_label = 'negative' THEN 1 ELSE 0 END)) * 1.0 / COUNT(*), 4) AS net_sentiment_roberta
            FROM company_enriched_view
            GROUP BY company, year, quarter, year_quarter
            ORDER BY company, year, quarter
        """
        df_quarterly = self.conn.execute(sql).fetchdf()
        logger.info(f"Generated quarterly mart: {len(df_quarterly):,} company-quarter aggregates.")
        return df_quarterly

    def query(self, sql: str) -> pd.DataFrame:
        """Executes a read-only SQL query against the DuckDB environment."""
        return self.conn.execute(sql).fetchdf()

    def build_and_save_all(
        self,
        df_enriched: Optional[pd.DataFrame] = None,
        df_mentions: Optional[pd.DataFrame] = None,
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Registers views, computes monthly/quarterly marts, and persists to Parquet."""
        self.register_tables(df_enriched, df_mentions)
        monthly_df = self.build_monthly_mart()
        quarterly_df = self.build_quarterly_mart()

        marts = {
            "mart_company_monthly_sentiment": monthly_df,
            "mart_quarterly_sentiment": quarterly_df,
        }

        if save:
            self.storage.save_parquet(monthly_df, "marts/mart_company_monthly_sentiment.parquet")
            self.storage.save_parquet(quarterly_df, "marts/mart_quarterly_sentiment.parquet")
            logger.info("Persisted monthly and quarterly marts to Parquet storage.")

        return marts
