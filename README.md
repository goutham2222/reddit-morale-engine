# Reddit Morale Engine: Organizational Culture & Employee Morale Analytics

An end-to-end, reproducible, production-grade intelligence engine designed to analyze enterprise organizational culture, executive leadership announcements, return-to-office (RTO) ultimatums, and restructuring waves (2020–2026, with backward compatibility to 2015) using Reddit workplace discourse.

The platform combines **DuckDB in-process SQL analytical marts** with **persistent Qdrant vector retrieval (MMR)** and a **LangGraph multi-agent research team** (`planner` → `sql_analyst` || `qdrant_retriever` → `verifier` → `synthesizer`), serving both Microsoft Foundry Hosted Agent endpoints on port `8088` and an interactive **Streamlit + Plotly** analytics frontend.

---

## 🌟 Key Architectural Innovations

1. **Dual-Grounding Synthesis:**
   Every executive finding pairs macro quantitative metrics (sample size, net sentiment ratios, VADER/RoBERTa/LLM scores) with verifiable qualitative citations (author role, subreddit, timestamp, key quote, thread permalink).
2. **Two-Stage Content Deduplication:**
   Stage 1 removes duplicate Reddit PKs (`post_id`, `comment_id`). Stage 2 hashes normalized content and collapses viral reposts (e.g. syndicated executive resignation letters) into a single canonical record while tracking `duplicate_count` to prevent synthetic volume inflation.
3. **Disambiguation & Strict Payload Isolation:**
   Multi-token contextual disambiguation prevents false-positive entity tagging (e.g., distinguishing `"Monthly Meta-Thread"` or device discussions from corporate `Meta` and `Apple`). Vector retrieval in Qdrant strictly enforces payload filters to eliminate cross-company semantic contamination.
4. **Econometric Difference-in-Differences (DiD):**
   Evaluates the causal effect of restructuring announcements on workplace trust by comparing a **Treated Cohort** ($\ge 1,000$ layoffs) against an empirically curated **Control Cohort** (headcount resilience / zero major layoffs).
5. **Anti-Hallucination & Sample Sufficiency Verification:**
   The `verifier` node strictly halts unsupported assertions, demanding empirical sample sizes ($N \ge 30$) and checking citation fidelity against canonical records before allowing synthesis.

---

## 🏗️ Architecture & Pipeline Flow

```
[Raw Sources: PRAW API / BrightData Scrapes / Azure Blob Storage]
                               │
                               ▼
        [Stage 1: Standardization & Zero Title Leakage]
           • Base-36 ID regex sanitization (^t[13]_)
           • Pure comment text (no parent title pollution)
           • Schema validation with Pandera
                               │
                               ▼
        [Stage 2: Tagging, Disambiguation & Deduplication]
           • Subreddit culture priors & workplace lexicon match
           • Disambiguation against consumer/meta false positives
           • Two-stage deduplication tracking duplicate_count
                               │
                               ▼
        [Stage 3: Multi-Tier Sentiment & Cultural Enrichment]
           • Tier 1: VADER compound score [-1.0, 1.0]
           • Tier 2: CardiffNLP Twitter-RoBERTa (512 sliding window)
           • Tier 3: Gemini 2.5 Flash (Pydantic schema + checkpoint/resume)
                               │
                               ▼
        [Stage 4: DuckDB Parquet Marts & Econometric Modeling]
           • In-process SQL over Snappy-compressed Parquet views
           • Monthly & quarterly net sentiment aggregations
           • Event study pre/post windows (T-3 to T+3 months)
           • Difference-in-Differences (DiD) estimator: β_DiD
                               │
                               ▼
        [Stage 5: Persistent Qdrant & LangGraph Multi-Agent Engine]
           • 768-dimensional normalized embeddings (Cosine distance)
           • Payload-filtered Maximal Marginal Relevance (MMR)
           • LangGraph: Planner -> SQL Analyst || Retriever -> Verifier -> Synthesizer
                               │
                               ▼
        [Stage 6: Microsoft Foundry Hosted Agent & Streamlit UI]
           • FastAPI server on port 8088 with SSE token streaming
           • Streamlit + Plotly executive dashboard (port 8501)
           • Docker Compose & Azure Container Apps deployment
```

---

## 📁 Repository Structure

```
reddit-morale-engine/
├── configs/
│   ├── companies.yaml              # ~30 target enterprise & tech companies
│   ├── subreddits.yaml             # Workplace subreddits with culture priors
│   ├── workplace_lexicon.yaml      # Categorized culture, morale, leadership terms
│   ├── leadership_events.yaml      # Event study definitions (Jassy, Musk, Zuckerberg, etc.)
│   └── did_cohorts.yaml            # Treated vs. Control cohorts for DiD analysis
├── data/
│   ├── parquet/                    # Snappy Parquet data lake (local & Azure fallback)
│   ├── checkpoints/                # JSONL checkpoint files for resilient enrichment
│   └── qdrant/                     # Persistent local Qdrant vector database
├── frontend/
│   └── app.py                      # Interactive Streamlit + Plotly visual dashboard
├── notebooks/                      # 4 Thin Visual Audit & Exploration Notebooks
│   ├── 01_audit_ingestion_and_tagging.ipynb
│   ├── 02_audit_sentiment_models.ipynb
│   ├── 03_explore_marts_and_did.ipynb
│   └── 04_eval_langgraph_agent.ipynb
├── scripts/
│   ├── build_marts.py              # CLI DuckDB marts materializer
│   ├── fetch_2026_data.py          # Deduplicating incremental 2026 PRAW fetcher
│   └── index_qdrant.py             # CLI Qdrant embedding & vector indexer
├── src/
│   ├── common/                     # Config, Azure/Local storage, text utils, logger
│   ├── ingest/                     # Standardizer, Pandera validator, PRAW/BrightData loaders
│   ├── tagging/                    # Relevance filter, disambiguator, two-stage deduplicator
│   ├── enrichment/                 # VADER, RoBERTa chunker, Gemini 2.5 Flash enricher
│   ├── analytics/                  # DuckDB marts, event studies, DiD econometric model
│   ├── rag/                        # 768-dim Embeddings, Qdrant indexer, MMR retriever
│   └── agents/                     # LangGraph state, tools, nodes, compiled graph, FastAPI server
├── tests/                          # 45 Comprehensive Unit & Integration Tests
├── Dockerfile.backend              # Hosted Agent Service container definition
├── Dockerfile.frontend             # Streamlit Dashboard container definition
├── docker-compose.yml              # Local multi-container orchestration
├── azure.yaml                      # Azure Developer CLI deployment specification
├── pyproject.toml                  # Project metadata, dependencies, and pytest configuration
└── .env.example                    # Template environment variables
```

---

## ⚡ Quickstart

### 1. Environment Setup

```bash
git clone https://github.com/goutham2222/reddit-morale-engine.git
cd reddit-morale-engine

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### 2. Configure Environment

Copy `.env.example` to `.env` and set your credentials:

```bash
cp .env.example .env
```

Key variables:
- `GEMINI_API_KEY`: API key for Gemini 2.5 Flash enrichment (optional; rule-based fallback active).
- `AZURE_STORAGE_CONNECTION_STRING`: Optional connection string for Azure Blob Storage (falls back to local `data/parquet`).

### 3. Run the Test Suite

Execute the 45-test verification suite:

```bash
pytest tests/
```

### 4. Build Analytical Marts & Index Vectors

```bash
# Materialize DuckDB analytical marts from Parquet
python scripts/build_marts.py

# Index records into persistent Qdrant vector database
python scripts/index_qdrant.py
```

### 5. Launch Hosted Agent Server & Streamlit Dashboard

**Option A: Local Processes**

```bash
# Terminal 1: Microsoft Foundry Hosted Agent Server (Port 8088)
python -m src.agents.server

# Terminal 2: Streamlit + Plotly Dashboard (Port 8501)
streamlit run frontend/app.py
```

**Option B: Docker Compose**

```bash
docker-compose up --build
```

Access the Streamlit Dashboard at `http://localhost:8501` and the Agent API at `http://localhost:8088/docs`.

---

## 🔬 Econometric Methodology: Difference-in-Differences (DiD)

The engine models the impact of restructuring announcements on employee trust via a two-period Difference-in-Differences estimator:

$$\beta_{DiD} = \left(\bar{Y}_{\text{Treated, Post}} - \bar{Y}_{\text{Treated, Pre}}\right) - \left(\bar{Y}_{\text{Control, Post}} - \bar{Y}_{\text{Control, Pre}}\right)$$

Where:
- $\bar{Y}$: Mean net sentiment ratio $\frac{\text{Positive} - \text{Negative}}{\text{Total}}$ or RoBERTa score.
- **Treated Cohort:** Firms executing $\ge 1,000$ layoffs (e.g. Amazon, Meta, Google, Microsoft, Twitter/X, Salesforce).
- **Control Cohort:** Resilient firms with zero or low restructuring shocks (e.g. Apple, Airbnb, Nvidia).
- **Parallel Trends:** Validated pre-intervention trends ensure baseline compatibility prior to the shock.

---

## 📓 Thin Visual Audit Notebooks

The `notebooks/` directory contains 4 inspection notebooks designed to verify intermediate outputs:
1. `01_audit_ingestion_and_tagging.ipynb`: Inspects canonical schemas, title isolation, subreddit priors, and Stage 2 deduplication.
2. `02_audit_sentiment_models.ipynb`: Verifies VADER distributions, RoBERTa chunking, and Gemini Pydantic validation.
3. `03_explore_marts_and_did.ipynb`: Runs DuckDB SQL queries, generates Plotly event studies, and estimates DiD treatment effects.
4. `04_eval_langgraph_agent.ipynb`: Runs multi-agent queries, tests payload-filtered MMR, and inspects agent graph execution traces.

---

## 🔒 License

MIT License. Designed and engineered for organizational culture research and executive workforce intelligence.
