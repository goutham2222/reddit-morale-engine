"""Difference-in-Differences (DiD) econometric modeling of 2022-2026 tech layoffs and corporate restructuring."""

from typing import Dict, List, Optional
import pandas as pd

from src.analytics.duckdb_marts import DuckDBMartsManager
from src.common.azure_storage import StorageManager
from src.common.config import EngineConfig
from src.common.logger import get_logger

logger = get_logger("did_analysis")


class DiDAnalyzer:
    """Estimates the causal impact of mass layoffs on employee morale, trust, and burnout using DiD."""

    def __init__(
        self,
        marts_manager: Optional[DuckDBMartsManager] = None,
        config: Optional[EngineConfig] = None,
    ):
        self.config = config or EngineConfig.load()
        self.marts = marts_manager or DuckDBMartsManager(self.config)
        self.storage = StorageManager(self.config)
        self.did_cfg = self.config.did_cohorts.get("did_study", {})

        # Extract cohorts
        cohorts = self.did_cfg.get("cohorts", {})
        treated_list = cohorts.get("treated", {}).get("companies", [])
        self.treated_companies = [c["name"] if isinstance(c, dict) else c for c in treated_list]

        control_list = cohorts.get("control", {}).get("companies", [])
        self.control_companies = [c["name"] if isinstance(c, dict) else c for c in control_list]

        # Extract temporal windows
        self.pre_start = self.did_cfg.get("pre_period", {}).get("start", "2021-07")
        self.pre_end = self.did_cfg.get("pre_period", {}).get("end", "2022-06")

        self.shock_start = self.did_cfg.get("shock_period", {}).get("start", "2022-07")
        self.shock_end = self.did_cfg.get("shock_period", {}).get("end", "2023-06")

        self.post_start = self.did_cfg.get("post_period", {}).get("start", "2023-07")
        self.post_end = self.did_cfg.get("post_period", {}).get("end", "2024-12")

        self.ext_start = self.did_cfg.get("extended_period", {}).get("start", "2025-01")
        self.ext_end = self.did_cfg.get("extended_period", {}).get("end", "2026-12")

    def run_cohort_aggregates(self) -> pd.DataFrame:
        """Computes cohort metrics across Pre, Shock, Post, and Extended AI-era restructuring periods."""
        treated_sql = ", ".join(f"'{c}'" for c in self.treated_companies)
        control_sql = ", ".join(f"'{c}'" for c in self.control_companies)

        sql = f"""
            WITH categorized AS (
                SELECT 
                    record_id,
                    company,
                    year_month,
                    vader_score,
                    roberta_score,
                    llm_sentiment_score,
                    culture_intensity,
                    CASE 
                        WHEN company IN ({treated_sql}) THEN 'Treated (Mass Layoffs)'
                        WHEN company IN ({control_sql}) THEN 'Control (Resilient)'
                        ELSE 'Other'
                    END AS cohort,
                    CASE 
                        WHEN year_month BETWEEN '{self.pre_start}' AND '{self.pre_end}' THEN '1_Pre (2021Q3-2022Q2)'
                        WHEN year_month BETWEEN '{self.shock_start}' AND '{self.shock_end}' THEN '2_Shock (2022Q3-2023Q2)'
                        WHEN year_month BETWEEN '{self.post_start}' AND '{self.post_end}' THEN '3_Post (2023Q3-2024Q4)'
                        WHEN year_month BETWEEN '{self.ext_start}' AND '{self.ext_end}' THEN '4_Extended (2025-2026)'
                        ELSE 'Outside'
                    END AS study_period
                FROM company_enriched_view
                WHERE company IN ({treated_sql}, {control_sql})
            )
            SELECT 
                cohort,
                study_period,
                COUNT(*) AS sample_size,
                ROUND(AVG(vader_score), 4) AS avg_vader,
                ROUND(AVG(roberta_score), 4) AS avg_roberta,
                ROUND(AVG(llm_sentiment_score), 4) AS avg_llm,
                ROUND(AVG(culture_intensity), 2) AS avg_intensity
            FROM categorized
            WHERE study_period != 'Outside'
            GROUP BY cohort, study_period
            ORDER BY cohort, study_period
        """
        return self.marts.query(sql)

    def calculate_did_estimator(self, score_col: str = "avg_roberta") -> Dict[str, any]:
        """Calculates the Difference-in-Differences statistical matrix for pre vs. post restructuring."""
        df_agg = self.run_cohort_aggregates()
        if len(df_agg) == 0:
            return {"did_estimator": 0.0, "status": "insufficient_data"}

        pivot = df_agg.pivot(index="cohort", columns="study_period", values=score_col)

        pre_col = "1_Pre (2021Q3-2022Q2)"
        post_col = "3_Post (2023Q3-2024Q4)"
        shock_col = "2_Shock (2022Q3-2023Q2)"
        ext_col = "4_Extended (2025-2026)"

        treated_key = "Treated (Mass Layoffs)"
        control_key = "Control (Resilient)"

        def get_val(cohort_name: str, period_name: str) -> float:
            try:
                v = pivot.loc[cohort_name, period_name]
                return float(v) if pd.notnull(v) else 0.0
            except KeyError:
                return 0.0

        # Pre vs Post DiD
        treated_pre = get_val(treated_key, pre_col)
        treated_post = get_val(treated_key, post_col)
        control_pre = get_val(control_key, pre_col)
        control_post = get_val(control_key, post_col)

        delta_treated = round(treated_post - treated_pre, 4)
        delta_control = round(control_post - control_pre, 4)
        did_estimator = round(delta_treated - delta_control, 4)

        # Pre vs Shock DiD
        treated_shock = get_val(treated_key, shock_col)
        control_shock = get_val(control_key, shock_col)
        shock_delta_treated = round(treated_shock - treated_pre, 4)
        shock_delta_control = round(control_shock - control_pre, 4)
        shock_did = round(shock_delta_treated - shock_delta_control, 4)

        return {
            "metric": score_col,
            "treated_pre": treated_pre,
            "treated_post": treated_post,
            "delta_treated": delta_treated,
            "control_pre": control_pre,
            "control_post": control_post,
            "delta_control": delta_control,
            "did_post_vs_pre": did_estimator,
            "shock_did": shock_did,
            "cohort_matrix": pivot.to_dict(),
        }

    def save_did_mart(self) -> pd.DataFrame:
        """Materializes DiD cohort aggregates into marts/mart_did_layoffs_2022_2026.parquet."""
        df_agg = self.run_cohort_aggregates()
        if len(df_agg) > 0:
            self.storage.save_parquet(df_agg, "marts/mart_did_layoffs_2022_2026.parquet")
            logger.info("Saved mart_did_layoffs_2022_2026.parquet.")
        return df_agg
