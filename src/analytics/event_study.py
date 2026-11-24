"""Event study pre/post analyzer for leadership controversies, ultimatums, and corporate mandates."""

from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from dateutil.relativedelta import relativedelta

from src.analytics.duckdb_marts import DuckDBMartsManager
from src.common.azure_storage import StorageManager
from src.common.config import EngineConfig
from src.common.logger import get_logger

logger = get_logger("event_study")


def shift_month(year_month: str, delta_months: int) -> str:
    """Adds or subtracts months from a 'YYYY-MM' string."""
    dt = datetime.strptime(year_month, "%Y-%m")
    shifted = dt + relativedelta(months=delta_months)
    return shifted.strftime("%Y-%m")


class EventStudyAnalyzer:
    """Analyzes employee morale and cultural shifts across T-3 to T+3 parametric event windows."""

    def __init__(
        self,
        marts_manager: Optional[DuckDBMartsManager] = None,
        config: Optional[EngineConfig] = None,
    ):
        self.config = config or EngineConfig.load()
        self.marts = marts_manager or DuckDBMartsManager(self.config)
        self.storage = StorageManager(self.config)
        self.events_cfg = self.config.leadership_events.get("events", {})

    def analyze_event(self, event_key: str, event_meta: Dict) -> Dict:
        """Executes windowed pre/post analysis for a single leadership announcement or mandate."""
        company = event_meta.get("company", "")
        anchor = event_meta.get("anchor_month", "")
        pre_m = event_meta.get("window_pre_months", 3)
        post_m = event_meta.get("window_post_months", 3)

        pre_start = shift_month(anchor, -pre_m)
        pre_end = shift_month(anchor, -1)
        post_start = shift_month(anchor, 1)
        post_end = shift_month(anchor, post_m)

        sql = f"""
            WITH windowed AS (
                SELECT 
                    record_id,
                    year_month,
                    vader_score,
                    roberta_score,
                    llm_sentiment_score,
                    culture_intensity,
                    CASE 
                        WHEN year_month BETWEEN '{pre_start}' AND '{pre_end}' THEN 'pre'
                        WHEN year_month = '{anchor}' THEN 'event'
                        WHEN year_month BETWEEN '{post_start}' AND '{post_end}' THEN 'post'
                        ELSE 'outside'
                    END AS window_phase
                FROM company_enriched_view
                WHERE company = '{company}'
                  AND year_month BETWEEN '{pre_start}' AND '{post_end}'
            )
            SELECT 
                window_phase,
                COUNT(*) AS record_count,
                ROUND(AVG(vader_score), 4) AS avg_vader,
                ROUND(AVG(roberta_score), 4) AS avg_roberta,
                ROUND(AVG(llm_sentiment_score), 4) AS avg_llm,
                ROUND(AVG(culture_intensity), 2) AS avg_intensity
            FROM windowed
            GROUP BY window_phase
        """

        df_res = self.marts.query(sql).set_index("window_phase")

        # Extract pre metrics
        pre_records = int(df_res.loc["pre", "record_count"]) if "pre" in df_res.index else 0
        pre_vader = float(df_res.loc["pre", "avg_vader"]) if "pre" in df_res.index else 0.0
        pre_roberta = float(df_res.loc["pre", "avg_roberta"]) if "pre" in df_res.index else 0.0
        pre_llm = float(df_res.loc["pre", "avg_llm"]) if "pre" in df_res.index else 0.0
        pre_intensity = float(df_res.loc["pre", "avg_intensity"]) if "pre" in df_res.index else 0.0

        # Extract post metrics
        post_records = int(df_res.loc["post", "record_count"]) if "post" in df_res.index else 0
        post_vader = float(df_res.loc["post", "avg_vader"]) if "post" in df_res.index else 0.0
        post_roberta = float(df_res.loc["post", "avg_roberta"]) if "post" in df_res.index else 0.0
        post_llm = float(df_res.loc["post", "avg_llm"]) if "post" in df_res.index else 0.0
        post_intensity = float(df_res.loc["post", "avg_intensity"]) if "post" in df_res.index else 0.0

        # Event month metrics
        event_records = int(df_res.loc["event", "record_count"]) if "event" in df_res.index else 0

        # Calculate deltas (Post - Pre)
        delta_vader = round(post_vader - pre_vader, 4) if pre_records and post_records else 0.0
        delta_roberta = round(post_roberta - pre_roberta, 4) if pre_records and post_records else 0.0
        delta_llm = round(post_llm - pre_llm, 4) if pre_records and post_records else 0.0
        delta_intensity = round(post_intensity - pre_intensity, 2) if pre_records and post_records else 0.0

        vol_pct = (
            round(((post_records - pre_records) / max(1, pre_records)) * 100.0, 1)
            if pre_records
            else 0.0
        )

        return {
            "event_key": event_key,
            "company": company,
            "leader": event_meta.get("leader", ""),
            "title": event_meta.get("title", ""),
            "anchor_month": anchor,
            "pre_window": f"{pre_start} to {pre_end}",
            "post_window": f"{post_start} to {post_end}",
            "pre_records": pre_records,
            "event_records": event_records,
            "post_records": post_records,
            "volume_change_pct": vol_pct,
            "pre_avg_vader": pre_vader,
            "post_avg_vader": post_vader,
            "delta_vader": delta_vader,
            "pre_avg_roberta": pre_roberta,
            "post_avg_roberta": post_roberta,
            "delta_roberta": delta_roberta,
            "pre_avg_llm": pre_llm,
            "post_avg_llm": post_llm,
            "delta_llm": delta_llm,
            "pre_avg_intensity": pre_intensity,
            "post_avg_intensity": post_intensity,
            "delta_intensity": delta_intensity,
        }

    def run_all_events(self, save: bool = True) -> pd.DataFrame:
        """Analyzes all events registered in configs/leadership_events.yaml."""
        logger.info(f"Running event-study analysis for {len(self.events_cfg)} corporate events...")
        results = []
        for key, meta in self.events_cfg.items():
            res = self.analyze_event(key, meta)
            results.append(res)

        df_events = pd.DataFrame(results)
        if save and len(df_events) > 0:
            self.storage.save_parquet(df_events, "marts/mart_event_study_pre_post.parquet")
            logger.info("Persisted mart_event_study_pre_post.parquet.")

        return df_events
