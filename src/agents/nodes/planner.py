"""Planner node: deconstructs research queries into structured subtasks, filters, and execution intent."""

import re
from typing import Dict, List
from src.agents.state import ResearchAgentState
from src.common.config import EngineConfig
from src.common.logger import get_logger

logger = get_logger("planner_node")


def planner_node(state: ResearchAgentState) -> Dict:
    """Analyzes the user research query and extracts structured metadata constraints."""
    query = state.get("query", "")
    config = EngineConfig.load()
    companies_cfg = config.companies.get("companies", {})
    lower_query = query.lower()

    # 1. Identify Target Companies
    target_companies = []
    for company_name, meta in companies_cfg.items():
        aliases = meta.get("aliases", [])
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias.lower())}\b", lower_query):
                target_companies.append(company_name)
                break

    # 2. Identify Target Themes
    target_themes = []
    if any(w in lower_query for w in ["rto", "return to office", "in-office", "badge"]):
        target_themes.append("remote_rto_policy")
    if any(w in lower_query for w in ["layoff", "laid off", "job cuts", "severance", "restructuring"]):
        target_themes.append("career_progression_promotion")
    if any(w in lower_query for w in ["burnout", "overtime", "oncall", "pagerduty"]):
        target_themes.append("burnout_overtime_oncall")
    if any(w in lower_query for w in ["pip", "stack ranking", "quota", "micromanage"]):
        target_themes.append("micromanagement")
    if any(w in lower_query for w in ["leadership", "jassy", "zuck", "musk", "pichai", "ceo", "management"]):
        target_themes.append("leadership_management_quality")
    if any(w in lower_query for w in ["dei", "diversity", "inclusion"]):
        target_themes.append("diversity_inclusion_belonging")

    # 3. Determine Execution Intent
    is_quant = any(w in lower_query for w in [
        "trend", "score", "average", "did", "difference", "delta", "statistics",
        "pre/post", "event study", "volume", "how many", "count", "metrics"
    ])
    is_qual = any(w in lower_query for w in [
        "quote", "feel", "why", "voices", "perspective", "lived experience",
        "stories", "what did they say", "reactions"
    ])

    if is_quant and not is_qual:
        intent = "quantitative"
    elif is_qual and not is_quant:
        intent = "qualitative"
    else:
        intent = "mixed"

    # 4. Extract Date / Temporal Hints
    date_range = None
    year_match = re.search(r"\b(20[12][0-9])\b", query)
    month_match = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", lower_query)

    if year_match:
        yr = year_match.group(1)
        if month_match:
            month_map = {
                "january": "01", "february": "02", "march": "03", "april": "04",
                "may": "05", "june": "06", "july": "07", "august": "08",
                "september": "09", "october": "10", "november": "11", "december": "12"
            }
            mo = month_map[month_match.group(1)]
            date_range = {"start": f"{yr}-{mo}", "end": f"{yr}-{mo}"}
        else:
            date_range = {"start": f"{yr}-01", "end": f"{yr}-12"}

    plan_notes = (
        f"Intent: {intent} | Companies: {target_companies or 'Industry-wide'} | "
        f"Themes: {target_themes} | Date Range: {date_range}"
    )
    logger.info(f"Planner finished: {plan_notes}")

    return {
        "intent": intent,
        "target_companies": target_companies,
        "target_themes": target_themes,
        "target_subreddits": [],
        "date_range": date_range,
        "plan_notes": plan_notes,
    }
