"""Analytics modules: DuckDB analytical views, event-study pre/post models, and DiD restructuring analysis."""

from src.analytics.duckdb_marts import DuckDBMartsManager
from src.analytics.event_study import EventStudyAnalyzer
from src.analytics.did_analysis import DiDAnalyzer

__all__ = ["DuckDBMartsManager", "EventStudyAnalyzer", "DiDAnalyzer"]
