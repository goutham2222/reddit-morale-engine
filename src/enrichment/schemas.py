"""Pydantic schemas for structured LLM culture enrichment output and unified sentiment representations."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class CompanyStance(BaseModel):
    """Per-company sentiment breakdown when multiple firms are referenced."""
    company: str = Field(description="Canonical company name (e.g. Amazon, Google, Meta)")
    score: float = Field(ge=-1.0, le=1.0, description="Sentiment score normalized to [-1.0, +1.0]")
    stance: Literal["supportive", "critical", "neutral", "mixed"] = Field(
        description="Employee/observer stance towards this specific firm"
    )


class RedditCultureEnrichment(BaseModel):
    """Structured culture and morale schema extracted by Gemini 2.5 Flash."""
    is_culture_related: bool = Field(
        description="Whether the text discusses workplace culture, employee morale, leadership quality, or working conditions"
    )
    culture_intensity: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Intensity of cultural discussion: 1 (passing mention) to 5 (deep lived personal critique/experience)"
    )
    actor_role: Literal[
        "current_employee",
        "former_employee",
        "applicant",
        "manager_lead",
        "other",
        "unknown"
    ] = Field(
        default="unknown",
        description="Inferred professional relationship of the author to the discussed firm(s)"
    )
    recommendation: Literal[
        "join",
        "stay",
        "leave",
        "avoid",
        "mixed",
        "unknown"
    ] = Field(
        default="unknown",
        description="Explicit or implicit career recommendation to other candidates or peers"
    )
    themes: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Up to 5 themes selected strictly from the 15-theme taxonomy"
    )
    entity_sentiments: List[CompanyStance] = Field(
        default_factory=list,
        description="Breakdown of sentiments per discussed company"
    )
    negation_near_theme: bool = Field(
        default=False,
        description="True if negation words (e.g., 'not', 'never', 'hardly') modify or invert the core cultural theme"
    )
    key_quote: Optional[str] = Field(
        default=None,
        max_length=200,
        description="A concise verbatim excerpt (<= 200 chars) capturing the central morale or leadership argument"
    )
    sentiment_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Overall LLM sentiment score normalized to [-1.0, +1.0]"
    )
    sentiment_label: Literal["positive", "negative", "neutral"] = Field(
        default="neutral",
        description="Overall sentiment classification based on shared thresholds"
    )
