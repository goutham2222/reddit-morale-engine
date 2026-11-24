"""SQL analyst node: formulates and executes analytical queries against DuckDB marts."""

from typing import Dict, List
from src.agents.state import ResearchAgentState
from src.agents.tools import DuckDBSQLTool
from src.common.logger import get_logger

logger = get_logger("sql_analyst_node")


def sql_analyst_node(state: ResearchAgentState) -> Dict:
    """Executes analytical DuckDB queries for quantitative metrics, pre/post deltas, and cohort comparisons."""
    companies = state.get("target_companies", [])
    date_range = state.get("date_range")
    query = state.get("query", "").lower()
    tool = DuckDBSQLTool()

    # Case 1: Difference-in-Differences / Layoffs Restructuring Query
    if "did" in query or "difference in difference" in query or "treatment" in query:
        sql = """
            SELECT 
                cohort,
                study_period,
                sample_size,
                avg_roberta,
                avg_llm,
                avg_intensity
            FROM read_parquet('data/parquet/marts/mart_did_layoffs_2022_2026.parquet')
            ORDER BY cohort, study_period
        """
    # Case 2: Event Study / Pre-Post Controversy Analysis
    elif any(w in query for w in ["ultimatum", "event study", "pre/post", "before and after", "announcement"]):
        comp_filter = f"WHERE company = '{companies[0]}'" if companies else ""
        sql = f"""
            SELECT 
                company,
                anchor_month,
                title,
                pre_records,
                post_records,
                delta_roberta,
                delta_llm,
                volume_change_pct
            FROM read_parquet('data/parquet/marts/mart_event_study_pre_post.parquet')
            {comp_filter}
        """
    # Case 3: Monthly Company Sentiment Trajectory
    elif companies:
        comp_str = ", ".join(f"'{c}'" for c in companies)
        date_filter = ""
        if date_range:
            start_ym = date_range.get("start", "")
            end_ym = date_range.get("end", "")
            date_filter = f"AND year_month BETWEEN '{start_ym}' AND '{end_ym}'"

        sql = f"""
            SELECT 
                company,
                year_month,
                total_records,
                avg_vader_score,
                avg_roberta_score,
                avg_llm_score,
                net_sentiment_roberta,
                avg_culture_intensity
            FROM read_parquet('data/parquet/marts/mart_company_monthly_sentiment.parquet')
            WHERE company IN ({comp_str}) {date_filter}
            ORDER BY company, year_month
        """
    # Case 4: General Industry Aggregate
    else:
        sql = """
            SELECT 
                company,
                SUM(total_records) AS total_mentions,
                ROUND(AVG(avg_roberta_score), 4) AS overall_roberta_sentiment,
                ROUND(AVG(net_sentiment_roberta), 4) AS overall_net_sentiment
            FROM read_parquet('data/parquet/marts/mart_company_monthly_sentiment.parquet')
            GROUP BY company
            ORDER BY total_mentions DESC
            LIMIT 10
        """

    try:
        results = tool.execute_sql(sql)
    except Exception as e:
        logger.warning(f"Fallback to company_enriched_view direct SQL query: {e}")
        # Fallback query directly on view if materialized parquet mart is not yet on disk
        if companies:
            comp_name = companies[0]
            fallback_sql = f"""
                SELECT 
                    company,
                    COUNT(*) AS total_records,
                    ROUND(AVG(vader_score), 4) AS avg_vader_score,
                    ROUND(AVG(roberta_score), 4) AS avg_roberta_score,
                    ROUND(AVG(llm_sentiment_score), 4) AS avg_llm_score
                FROM company_enriched_view
                WHERE company = '{comp_name}'
                GROUP BY company
            """
            results = tool.execute_sql(fallback_sql)
            sql = fallback_sql
        else:
            results = []

    logger.info(f"SQL Analyst retrieved {len(results)} rows.")
    return {
        "sql_query": sql,
        "sql_results": results,
    }
