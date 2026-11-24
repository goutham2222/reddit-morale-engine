"""Tagging modules: workplace relevance filtering, entity disambiguation, and two-stage deduplication."""

from src.tagging.relevance_filter import RelevanceFilter
from src.tagging.entity_disambiguator import EntityDisambiguator
from src.tagging.deduplicator import TwoStageDeduplicator
from src.tagging.pipeline import TaggingPipeline

__all__ = ["RelevanceFilter", "EntityDisambiguator", "TwoStageDeduplicator", "TaggingPipeline"]
