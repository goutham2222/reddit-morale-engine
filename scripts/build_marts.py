"""CLI script to build and materialize all DuckDB analytical marts and econometric models."""

import argparse
from src.analytics.did_analysis import DiDAnalyzer
from src.analytics.duckdb_marts import DuckDBMartsManager
from src.analytics.event_study import EventStudyAnalyzer
from src.common.config import EngineConfig
from src.common.logger import get_logger

logger = get_logger("build_marts")


def build_all_analytical_marts():
    """Generates all DuckDB views, monthly/quarterly marts, event study, and DiD models."""
    config = EngineConfig.load()
    logger.info("Initializing DuckDB analytical engine...")

    marts_mgr = DuckDBMartsManager(config)

    # 1. Build and save monthly and quarterly marts
    logger.info("Building Monthly and Quarterly marts...")
    marts_mgr.build_and_save_all(save=True)

    # 2. Build and save Event Study models
    logger.info("Building Event Study pre/post controversy models...")
    event_analyzer = EventStudyAnalyzer(marts_manager=marts_mgr, config=config)
    event_analyzer.run_all_events(save=True)

    # 3. Build and save Difference-in-Differences models
    logger.info("Building Difference-in-Differences restructuring models...")
    did_analyzer = DiDAnalyzer(marts_manager=marts_mgr, config=config)
    did_analyzer.save_did_mart()

    logger.info("All analytical Parquet marts materialized successfully in data/parquet/marts/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and materialize DuckDB analytical marts")
    args = parser.parse_args()
    build_all_analytical_marts()
