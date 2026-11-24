"""Tier 1: Fast deterministic lexicon-based sentiment analysis using VADER."""

from typing import Dict, Optional, Tuple
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.common.logger import get_logger

logger = get_logger("vader_analyzer")


def classify_sentiment(score: float, threshold: float = 0.20) -> str:
    """Applies unified sentiment thresholding across [-1.0, +1.0]."""
    if score >= threshold:
        return "positive"
    elif score <= -threshold:
        return "negative"
    return "neutral"


class VaderAnalyzer:
    """Lexicon-based sentiment intensity analyzer normalized to [-1.0, +1.0]."""

    def __init__(self, threshold: float = 0.20):
        self.analyzer = SentimentIntensityAnalyzer()
        self.threshold = threshold

    def analyze_text(self, text: str) -> Dict[str, any]:
        """Calculates VADER compound sentiment score and unified classification label."""
        cleaned = (text or "").strip()
        if not cleaned:
            return {"vader_score": 0.0, "vader_label": "neutral"}

        scores = self.analyzer.polarity_scores(cleaned)
        compound = float(scores["compound"])
        label = classify_sentiment(compound, self.threshold)

        return {
            "vader_score": round(compound, 4),
            "vader_label": label,
        }

    def enrich_dataframe(
        self, df: pd.DataFrame, text_column: str = "clean_text"
    ) -> pd.DataFrame:
        """Enriches a DataFrame with vader_score and vader_label columns."""
        logger.info(f"Running VADER Tier 1 enrichment on {len(df):,} records...")
        results = [self.analyze_text(str(t)) for t in df[text_column]]
        res_df = pd.DataFrame(results)

        df_enriched = df.copy()
        df_enriched["vader_score"] = res_df["vader_score"].values
        df_enriched["vader_label"] = res_df["vader_label"].values

        counts = df_enriched["vader_label"].value_counts().to_dict()
        logger.info(f"VADER Enrichment Complete: {counts}")
        return df_enriched
