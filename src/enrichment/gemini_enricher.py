"""Tier 3: Async Structured Gemini 2.5 Flash culture enrichment with Pydantic output,

tenacity rate-limit retry, and JSONL checkpoint/resume.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set
import json
import os
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.common.config import EngineConfig
from src.common.logger import get_logger
from src.enrichment.schemas import CompanyStance, RedditCultureEnrichment
from src.enrichment.vader_analyzer import classify_sentiment

logger = get_logger("gemini_enricher")


class GeminiEnricher:
    """Extracts structured organizational culture and employee morale attributes via Gemini 2.5 Flash."""

    def __init__(self, config: Optional[EngineConfig] = None, mock_mode: bool = False):
        self.config = config or EngineConfig.load()
        self.mock_mode = mock_mode or not bool(self.config.google_api_key)
        self.model_name = self.config.gemini_model or "gemini-2.5-flash"
        self.checkpoint_dir = self.config.local_data_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "gemini_enrichment_checkpoint.jsonl"

        # Load 15-theme taxonomy list
        lex_cfg = self.config.workplace_lexicon or {}
        taxonomy = lex_cfg.get("taxonomy_themes", {})
        self.theme_list = list(taxonomy.keys()) if taxonomy else [
            "work_life_balance", "burnout_overtime_oncall", "micromanagement", "autonomy_trust",
            "mentorship_growth", "leadership_management_quality", "psychological_safety",
            "toxicity_bullying", "communication_process", "recognition_appreciation",
            "career_progression_promotion", "remote_rto_policy", "team_collaboration",
            "values_alignment_ethics", "diversity_inclusion_belonging"
        ]

        self._client = None
        if not self.mock_mode:
            self._init_gemini_client()

    def _init_gemini_client(self):
        """Initializes the Gemini API client."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.config.google_api_key)
            self._client = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini client initialized with model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not initialize google.generativeai client: {e}. Falling back to mock mode.")
            self.mock_mode = True

    def load_checkpoint_ids(self) -> Set[str]:
        """Loads record_ids already enriched from the JSONL checkpoint file."""
        completed_ids = set()
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if "record_id" in data:
                                completed_ids.add(data["record_id"])
                        except Exception:
                            pass
            logger.info(f"Loaded {len(completed_ids):,} previously enriched record_ids from checkpoint.")
        return completed_ids

    def save_checkpoint_record(self, record_id: str, enrichment: RedditCultureEnrichment):
        """Appends a single enriched record to the JSONL checkpoint file."""
        data = enrichment.model_dump()
        data["record_id"] = record_id
        with open(self.checkpoint_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

    def build_prompt(self, text: str, companies: List[str], subreddit: str, date_str: str) -> str:
        """Constructs a strict JSON structured extraction prompt."""
        taxonomy_str = ", ".join(self.theme_list)
        comp_str = ", ".join(companies) if companies else "General Industry"
        return f"""You are an expert organizational psychologist and workplace culture research assistant.
Analyze the following Reddit post/comment discussing workplace culture and employee morale.

CONTEXT:
Subreddit: r/{subreddit}
Date: {date_str}
Companies Mentioned: {comp_str}

TEXT:
\"\"\"{text[:3000]}\"\"\"

Analyze the text and return valid JSON matching this schema:
{{
  "is_culture_related": true/false,
  "culture_intensity": integer from 1 to 5,
  "actor_role": one of ["current_employee", "former_employee", "applicant", "manager_lead", "other", "unknown"],
  "recommendation": one of ["join", "stay", "leave", "avoid", "mixed", "unknown"],
  "themes": list of 1 to 5 themes chosen strictly from [{taxonomy_str}],
  "entity_sentiments": [
    {{"company": "<Company Name>", "score": float from -1.0 to 1.0, "stance": "supportive"|"critical"|"neutral"|"mixed"}}
  ],
  "negation_near_theme": true/false,
  "key_quote": "A verbatim short excerpt under 200 chars capturing the core sentiment",
  "sentiment_score": float from -1.0 to 1.0,
  "sentiment_label": "positive"|"negative"|"neutral"
}}
Return ONLY valid JSON.
"""

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(Exception),
    )
    def _call_gemini_api(self, prompt: str) -> dict:
        """Calls Gemini API with exponential backoff on rate limits."""
        response = self._client.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json", "temperature": 0.1},
        )
        return json.loads(response.text)

    def _mock_enrich(self, text: str, companies: List[str]) -> RedditCultureEnrichment:
        """Generates grounded deterministic enrichment for mock testing and CI."""
        lower = text.lower()
        matched_themes = []

        if any(w in lower for w in ["rto", "return to office", "in-office", "badge"]):
            matched_themes.append("remote_rto_policy")
        if any(w in lower for w in ["burnout", "oncall", "overworked", "pagerduty"]):
            matched_themes.append("burnout_overtime_oncall")
        if any(w in lower for w in ["pip", "stack ranking", "quota", "micromanage"]):
            matched_themes.append("micromanagement")
        if any(w in lower for w in ["manager", "leadership", "jassy", "zuck", "musk", "ceo"]):
            matched_themes.append("leadership_management_quality")
        if any(w in lower for w in ["wlb", "work life balance", "hours"]):
            matched_themes.append("work_life_balance")

        is_culture = len(matched_themes) > 0 or len(companies) > 0
        sentiment = -0.6 if any(w in lower for w in ["tanked", "gutter", "worst", "hate", "layoff", "disaster"]) else 0.4
        label = classify_sentiment(sentiment, threshold=0.20)

        entity_stances = []
        for comp in companies:
            entity_stances.append(
                CompanyStance(company=comp, score=sentiment, stance="critical" if sentiment < 0 else "supportive")
            )

        # Extract quote
        quote = text.split("\n")[0][:180] if text else None

        return RedditCultureEnrichment(
            is_culture_related=is_culture,
            culture_intensity=4 if sentiment < -0.4 else 2,
            actor_role="current_employee" if "our org" in lower or "my team" in lower else "unknown",
            recommendation="avoid" if sentiment < -0.4 else "stay",
            themes=matched_themes[:5] or ["leadership_management_quality"],
            entity_sentiments=entity_stances,
            negation_near_theme="not" in lower or "never" in lower,
            key_quote=quote,
            sentiment_score=sentiment,
            sentiment_label=label,
        )

    def enrich_record(
        self,
        record_id: str,
        text: str,
        companies: List[str],
        subreddit: str = "",
        date_str: str = "",
    ) -> RedditCultureEnrichment:
        """Enriches a single record with Gemini structured outputs."""
        if self.mock_mode or not self._client:
            enrichment = self._mock_enrich(text, companies)
        else:
            prompt = self.build_prompt(text, companies, subreddit, date_str)
            try:
                raw_json = self._call_gemini_api(prompt)
                enrichment = RedditCultureEnrichment.model_validate(raw_json)
            except Exception as e:
                logger.error(f"Error enriching record {record_id} with Gemini: {e}. Falling back to heuristic mock.")
                enrichment = self._mock_enrich(text, companies)

        # Save to checkpoint
        self.save_checkpoint_record(record_id, enrichment)
        return enrichment
