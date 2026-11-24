"""Reddit Morale Engine — Interactive Streamlit Dashboard.

Provides executive analytics, leadership event studies, Difference-in-Differences
econometric modeling, and LangGraph multi-agent research exploration.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.common.config import EngineConfig
from src.common.azure_storage import StorageManager
from src.analytics.duckdb_marts import DuckDBMartsManager
from src.analytics.event_study import EventStudyAnalyzer
from src.analytics.did_analysis import DiDAnalyzer
from src.rag.qdrant_indexer import QdrantIndexer
from src.rag.retriever import QdrantMMRRetriever
from src.agents.graph import create_research_graph, run_research_query


# -------------------------------------------------------------------------
# Page Setup
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Reddit Morale Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_engine():
    config = EngineConfig.load()
    storage = StorageManager(config)
    marts_mgr = DuckDBMartsManager(config)
    event_analyzer = EventStudyAnalyzer(config)
    did_analyzer = DiDAnalyzer(config)
    indexer = QdrantIndexer(config)
    retriever = QdrantMMRRetriever(config=config, indexer=indexer)
    research_graph = create_research_graph(config)
    return {
        "config": config,
        "storage": storage,
        "marts": marts_mgr,
        "event_analyzer": event_analyzer,
        "did_analyzer": did_analyzer,
        "indexer": indexer,
        "retriever": retriever,
        "graph": research_graph,
    }


engine = load_engine()

# -------------------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------------------
st.sidebar.title("Reddit Morale Engine")
st.sidebar.caption("Organizational Culture & Leadership Sentiment Analytics (2020–2026)")

mode = st.sidebar.radio(
    "Navigation",
    [
        "🤖 AI Research Analyst",
        "📈 Macro Morale Trends",
        "⚡ Leadership Event Studies",
        "🔬 Difference-in-Differences (DiD)",
        "🔍 Evidence Browser (Qdrant MMR)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Core Architecture:**\n"
    "- **DuckDB:** Parquet Analytical Marts\n"
    "- **Qdrant:** 768-dim MMR Vector Evidence\n"
    "- **LangGraph:** Multi-Agent Dual-Grounding\n"
    "- **Scope:** 2020–2026 Primary Focus"
)


# -------------------------------------------------------------------------
# View 1: AI Research Analyst (LangGraph)
# -------------------------------------------------------------------------
if mode == "🤖 AI Research Analyst":
    st.header("🤖 Multi-Agent Research Analyst")
    st.markdown(
        "Autonomous multi-agent research team combining **DuckDB SQL quantitative aggregations** "
        "with **payload-filtered Qdrant MMR qualitative citations**."
    )

    st.subheader("💡 Suggested Executive Research Queries")
    c1, c2, c3 = st.columns(3)
    preset_query = None

    if c1.button("🏢 Amazon 5-Day RTO (Sep 2024)", use_container_width=True):
        preset_query = "How did Andy Jassy's 5-day RTO announcement in September 2024 impact Amazon employee morale and resignation intent?"
    if c2.button("⚡ Meta 'Year of Efficiency' (Feb 2023)", use_container_width=True):
        preset_query = "What was the employee sentiment reaction to Mark Zuckerberg's 2023 Year of Efficiency and flattening middle management?"
    if c3.button("📉 2022–2026 Layoffs DiD Comparison", use_container_width=True):
        preset_query = "Analyze the causal impact of tech layoffs on employee trust using Difference-in-Differences comparing treated firms to control firms."

    query_input = st.text_input(
        "Enter your research inquiry:",
        value=preset_query or "How did return-to-office mandates affect employee trust across enterprise tech firms?",
    )

    if st.button("🚀 Run Multi-Agent Research", type="primary"):
        with st.spinner("Multi-agent team planning, querying DuckDB SQL marts, and retrieving Qdrant evidence..."):
            result = run_research_query(engine["graph"], query_input)

            # Executive Metrics Banner
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Target Companies", ", ".join(result.get("target_companies", [])) or "Cross-Industry")
            m2.metric("Sample Size Evaluated", f"{result.get('sample_size', 0):,} records")
            v_pass = result.get("verification_passed", False)
            m3.metric("Verification Status", "✅ Passed (N>=30)" if v_pass else "⚠️ Small Sample")
            m4.metric("Evidence Citations", f"{len(result.get('citations', []))} items")

            st.markdown("---")
            st.markdown("### 📋 Executive Research Report")
            st.markdown(result.get("final_report", "No report generated."))

            # Expandable Citations Drawer
            citations = result.get("citations", [])
            if citations:
                st.markdown("---")
                st.markdown("### 📚 Grounded Qualitative Evidence (Qdrant MMR)")
                for i, cite in enumerate(citations, 1):
                    with st.expander(f"[{i}] {cite.get('citation', 'Citation')} — Match Score: {cite.get('score', 0.0)}"):
                        st.markdown(f"**Author Role:** `{cite.get('role', 'unknown')}`")
                        st.markdown(f"**Subreddit:** `r/{cite.get('subreddit', 'unknown')}` | **Date:** `{cite.get('year_month', '')}`")
                        st.markdown(f"**Companies:** {cite.get('companies', [])} | **Themes:** {cite.get('themes', [])}")
                        st.markdown(f"> *\"{cite.get('key_quote', cite.get('text', ''))}\"*")
                        if cite.get("permalink"):
                            st.caption(f"[View Reddit Thread](https://reddit.com{cite['permalink']})")


# -------------------------------------------------------------------------
# View 2: Macro Morale Trends (2020–2026)
# -------------------------------------------------------------------------
elif mode == "📈 Macro Morale Trends":
    st.header("📈 Macro Sentiment & Morale Trajectories (2020–2026)")
    st.markdown("Aggregated monthly and quarterly net sentiment indices computed over Parquet marts.")

    try:
        engine["marts"].connect()
        monthly_df = engine["marts"].get_monthly_sentiment()
        engine["marts"].close()

        if not monthly_df.empty:
            companies = sorted(monthly_df["company"].dropna().unique().tolist())
            selected_companies = st.multiselect(
                "Filter Companies:",
                companies,
                default=companies[:5] if len(companies) >= 5 else companies,
            )

            filtered_df = monthly_df[monthly_df["company"].isin(selected_companies)]

            # Net Sentiment Trajectory Line Chart
            fig = px.line(
                filtered_df,
                x="year_month",
                y="net_sentiment_ratio",
                color="company",
                title="Monthly Net Sentiment Ratio: (Pos - Neg) / Total (2020–2026)",
                markers=True,
                labels={"year_month": "Month", "net_sentiment_ratio": "Net Sentiment Ratio"},
            )
            fig.add_hline(y=0.0, line_dash="dash", line_color="gray", annotation_text="Neutral Baseline")
            fig.update_layout(height=500, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Volume & Multi-tier breakdown
            c1, c2 = st.columns(2)
            with c1:
                vol_fig = px.bar(
                    filtered_df,
                    x="year_month",
                    y="total_volume",
                    color="company",
                    title="Monthly Discussion Volume (Mentions)",
                    labels={"year_month": "Month", "total_volume": "Volume"},
                )
                vol_fig.update_layout(height=400)
                st.plotly_chart(vol_fig, use_container_width=True)

            with c2:
                vader_fig = px.line(
                    filtered_df,
                    x="year_month",
                    y="vader_avg",
                    color="company",
                    title="VADER Mean Compound Score [-1.0, 1.0]",
                    labels={"year_month": "Month", "vader_avg": "VADER Compound"},
                )
                vader_fig.update_layout(height=400)
                st.plotly_chart(vader_fig, use_container_width=True)

            # Tabular Mart Explorer
            with st.expander("🔍 View Raw Monthly Mart Data"):
                st.dataframe(filtered_df, use_container_width=True)
        else:
            st.warning("No monthly mart data found in storage. Run `python scripts/build_marts.py` to materialize analytics.")
    except Exception as e:
        st.error(f"Error loading macro trends: {e}")


# -------------------------------------------------------------------------
# View 3: Leadership Event Studies
# -------------------------------------------------------------------------
elif mode == "⚡ Leadership Event Studies":
    st.header("⚡ Leadership Ultimatums & Policy Event Studies")
    st.markdown(
        "Quantifies immediate cultural shocks and employee morale shifts across a "
        "pre/post window ($T_{-3}$ to $T_{+3}$ months) surrounding major executive announcements."
    )

    c1, c2 = st.columns(2)
    with c1:
        company_choice = st.selectbox(
            "Select Company:",
            ["Amazon", "Meta", "Twitter", "Google", "Microsoft", "Apple", "Netflix"],
        )
    with c2:
        window_size = st.slider("Window Size (Months before/after)", min_value=1, max_value=6, value=3)

    if st.button("📊 Calculate Event Window Deltas", type="primary"):
        with st.spinner("Querying DuckDB marts for windowed sentiment shifts..."):
            res = engine["event_analyzer"].compute_event_study(
                company=company_choice,
                window_months=window_size,
            )

            ev = res.get("event", {})
            st.markdown(f"### Event: **{ev.get('title', 'Unknown Event')}**")
            st.caption(f"**Date:** {ev.get('date', 'N/A')} | **Leadership Figure:** {ev.get('leader', 'N/A')} | **Type:** `{ev.get('type', 'N/A')}`")

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Pre-Event Net Sentiment", f"{res.get('pre_mean_net_sentiment', 0.0):+.3f}")
            d2.metric("Post-Event Net Sentiment", f"{res.get('post_mean_net_sentiment', 0.0):+.3f}")
            delta = res.get("net_sentiment_delta", 0.0)
            d3.metric("Net Shift (Δ)", f"{delta:+.3f}", delta_color="normal" if delta >= 0 else "inverse")
            d4.metric("Window Records Evaluated", f"{res.get('sample_size', 0):,}")

            # Plot pre/post comparison
            window_df = pd.DataFrame(res.get("window_records", []))
            if not window_df.empty:
                w_fig = px.bar(
                    window_df,
                    x="year_month",
                    y="net_sentiment_ratio",
                    color="period",
                    color_discrete_map={"Pre-Event": "#1f77b4", "Post-Event": "#d62728"},
                    title=f"Monthly Trajectory Surrounding {ev.get('title', 'Event')}",
                    labels={"year_month": "Month", "net_sentiment_ratio": "Net Sentiment Ratio"},
                )
                w_fig.add_hline(y=0.0, line_dash="dash", line_color="gray")
                st.plotly_chart(w_fig, use_container_width=True)


# -------------------------------------------------------------------------
# View 4: Difference-in-Differences (DiD)
# -------------------------------------------------------------------------
elif mode == "🔬 Difference-in-Differences (DiD)":
    st.header("🔬 2022–2026 Tech Layoffs: Difference-in-Differences (DiD)")
    st.markdown(
        "Evaluates the causal effect of mass restructuring on workforce sentiment using an econometric "
        "Difference-in-Differences estimator comparing **Treated Firms** (mass layoffs $\\ge$ 1,000) against "
        "**Control Firms** (resilient workforce policies)."
    )

    st.latex(
        r"\beta_{DiD} = \left(\bar{Y}_{\text{Treated, Post}} - \bar{Y}_{\text{Treated, Pre}}\right) - "
        r"\left(\bar{Y}_{\text{Control, Post}} - \bar{Y}_{\text{Control, Pre}}\right)"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        interv_date = st.text_input("Intervention Month (YYYY-MM):", value="2022-11")
    with c2:
        pre_win = st.slider("Pre-Window (Months)", min_value=3, max_value=24, value=6)
    with c3:
        post_win = st.slider("Post-Window (Months)", min_value=3, max_value=24, value=6)

    if st.button("📐 Estimate DiD Econometric Treatment Effect", type="primary"):
        with st.spinner("Executing DiD estimator over DuckDB marts..."):
            did_res = engine["did_analyzer"].compute_did(
                intervention_start=interv_date,
                pre_window_months=pre_win,
                post_window_months=post_win,
            )

            # Treatment Effect Display
            b1, b2, b3, b4 = st.columns(4)
            beta = did_res.get("beta_did_net_sentiment", 0.0)
            b1.metric("DiD Estimator (β)", f"{beta:+.4f}", delta_color="normal" if beta >= 0 else "inverse")
            t_delta = did_res.get("treated_delta_net_sentiment", 0.0)
            b2.metric("Treated Cohort Δ", f"{t_delta:+.4f}")
            c_delta = did_res.get("control_delta_net_sentiment", 0.0)
            b3.metric("Control Cohort Δ", f"{c_delta:+.4f}")
            b4.metric("Total Records Evaluated", f"{did_res.get('total_sample_size', 0):,}")

            st.markdown("---")
            cohort_c1, cohort_c2 = st.columns(2)
            with cohort_c1:
                st.markdown("**Treated Cohort (Mass Layoffs):**")
                st.write(", ".join(did_res.get("treated_companies", [])))
            with cohort_c2:
                st.markdown("**Control Cohort (Resilient / Low-Impact):**")
                st.write(", ".join(did_res.get("control_companies", [])))

            # Trajectory Chart
            series_data = did_res.get("monthly_series", [])
            if series_data:
                s_df = pd.DataFrame(series_data)
                did_fig = px.line(
                    s_df,
                    x="year_month",
                    y="net_sentiment_ratio",
                    color="cohort",
                    title="Parallel Trends Trajectory: Treated vs. Control Cohorts",
                    markers=True,
                    labels={"year_month": "Month", "net_sentiment_ratio": "Net Sentiment Ratio"},
                )
                did_fig.add_vline(x=interv_date, line_dash="dash", line_color="red", annotation_text="Intervention Shock")
                st.plotly_chart(did_fig, use_container_width=True)


# -------------------------------------------------------------------------
# View 5: Evidence Browser (Qdrant MMR)
# -------------------------------------------------------------------------
elif mode == "🔍 Evidence Browser (Qdrant MMR)":
    st.header("🔍 Qualitative Evidence Browser (Persistent Qdrant MMR)")
    st.markdown(
        "Directly query the persistent Qdrant vector index (`reddit_culture_docs`) "
        "with strict metadata payload filtering to inspect frontline employee voices."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        search_query = st.text_input("Semantic Search Query:", value="toxic middle management and burnout")
    with c2:
        filter_company = st.text_input("Filter Company (Optional):", value="")
    with c3:
        filter_role = st.selectbox(
            "Filter Author Role:",
            ["All", "current_employee", "former_employee", "prospective_employee", "industry_observer"],
        )

    k_items = st.slider("Top-K Evidence Items", min_value=1, max_value=20, value=8)

    if st.button("🔎 Search Qdrant Index", type="primary"):
        with st.spinner("Performing payload-filtered MMR retrieval..."):
            companies_param = [filter_company.strip()] if filter_company.strip() else None
            role_param = [filter_role] if filter_role != "All" else None

            results = engine["retriever"].retrieve(
                query=search_query,
                companies=companies_param,
                actor_roles=role_param,
                top_k=k_items,
            )

            st.markdown(f"**Found {len(results)} diverse evidence chunks:**")
            for i, r in enumerate(results, 1):
                with st.container():
                    st.markdown(f"#### [{i}] {r.get('citation', '')} — Score: `{r.get('score', 0.0)}`")
                    st.markdown(f"**Company:** `{r.get('companies', [])}` | **Subreddit:** `r/{r.get('subreddit', '')}` | **Role:** `{r.get('role', '')}`")
                    st.markdown(f"> *\"{r.get('text', '')}\"*")
                    st.caption(f"Themes: {r.get('themes', [])} | Deduplication Reposts: {r.get('duplicate_count', 1)}")
                    st.markdown("---")
