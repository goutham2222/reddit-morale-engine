"""Microsoft Foundry Hosted Agent Server & FastAPI Engine on port 8088.

Exposes REST and Server-Sent Events (SSE) streaming endpoints for the LangGraph
multi-agent research orchestrator and analytical DuckDB marts.
"""

import json
import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional
from pydantic import BaseModel, Field

from src.common.config import EngineConfig
from src.common.logger import get_logger
from src.agents.graph import create_research_graph, run_research_query
from src.analytics.duckdb_marts import DuckDBMartsManager
from src.analytics.event_study import EventStudyAnalyzer
from src.analytics.did_analysis import DiDAnalyzer

logger = get_logger("server")

# -------------------------------------------------------------------------
# Request / Response Models
# -------------------------------------------------------------------------
class ResearchRequest(BaseModel):
    query: str = Field(..., description="Natural language question about organizational culture or morale")
    companies: Optional[List[str]] = Field(default=None, description="Optional target companies to filter")
    top_k_evidence: int = Field(default=8, ge=1, le=50, description="Max Qdrant MMR citations to return")


class ResearchResponse(BaseModel):
    query: str
    final_report: str
    target_companies: List[str]
    citations: List[Dict[str, Any]]
    sample_size: int
    verification_passed: bool
    sql_findings: Dict[str, Any]


class EventStudyRequest(BaseModel):
    company: str = Field(..., description="Target company, e.g. Amazon, Meta, Twitter")
    event_key: Optional[str] = Field(default=None, description="Event key from leadership_events.yaml")
    window_months: int = Field(default=3, ge=1, le=6)


class DiDRequest(BaseModel):
    intervention_start: str = Field(default="2022-11", description="Intervention start month (YYYY-MM)")
    pre_window_months: int = Field(default=6, ge=1, le=24)
    post_window_months: int = Field(default=6, ge=1, le=24)


# -------------------------------------------------------------------------
# FastAPI Application Factory
# -------------------------------------------------------------------------
def create_app(config: Optional[EngineConfig] = None):
    """Creates the FastAPI / Azure Foundry agent server."""
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError:
        logger.error("FastAPI is not installed. Please install fastapi and uvicorn.")
        raise

    cfg = config or EngineConfig.load()
    app = FastAPI(
        title="Reddit Morale Engine — Agent Service",
        description="Microsoft Foundry compatible multi-agent research and sentiment analytics service.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Lazy-load shared singletons
    research_graph = create_research_graph(cfg)
    marts_mgr = DuckDBMartsManager(cfg)
    event_analyzer = EventStudyAnalyzer(cfg)
    did_analyzer = DiDAnalyzer(cfg)

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "reddit-morale-engine",
            "port": 8088,
            "version": "0.1.0",
        }

    @app.get("/info")
    async def service_info():
        return {
            "name": "Reddit Morale Engine Agent Service",
            "architecture": "LangGraph (Planner -> Analyst || Retriever -> Verifier -> Synthesizer)",
            "storage": "DuckDB Parquet Marts + Persistent Qdrant (MMR)",
            "supported_years": "2015-2026 (Focus: 2020-2026)",
        }

    @app.post("/api/research", response_model=ResearchResponse)
    async def handle_research_query(req: ResearchRequest):
        """Executes the full multi-agent research workflow synchronously."""
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, run_research_query, research_graph, req.query
            )
            return ResearchResponse(
                query=req.query,
                final_report=result.get("final_report", ""),
                target_companies=result.get("target_companies", []),
                citations=result.get("citations", []),
                sample_size=result.get("sample_size", 0),
                verification_passed=result.get("verification_passed", False),
                sql_findings=result.get("sql_findings", {}),
            )
        except Exception as e:
            logger.error(f"Error handling research query: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/research/stream")
    async def handle_research_stream(req: ResearchRequest):
        """Streams Server-Sent Events (SSE) representing agent execution steps and tokens."""
        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                yield f"data: {json.dumps({'type': 'status', 'stage': 'planner', 'message': 'Extracting entities and planning query...'})}\n\n"
                await asyncio.sleep(0.1)

                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, run_research_query, research_graph, req.query
                )

                companies_str = ", ".join(result.get("target_companies", []))
                analyst_msg = f"Aggregated quantitative metrics for {companies_str}"
                yield f"data: {json.dumps({'type': 'status', 'stage': 'sql_analyst', 'message': analyst_msg})}\n\n"
                await asyncio.sleep(0.1)

                citations_cnt = len(result.get("citations", []))
                yield f"data: {json.dumps({'type': 'status', 'stage': 'qdrant_retriever', 'message': f'Retrieved {citations_cnt} employee citations.'})}\n\n"
                await asyncio.sleep(0.1)

                v_passed = result.get("verification_passed", False)
                s_size = result.get("sample_size", 0)
                verifier_msg = f"Verification passed: {v_passed} (N={s_size})"
                yield f"data: {json.dumps({'type': 'status', 'stage': 'verifier', 'message': verifier_msg})}\n\n"
                await asyncio.sleep(0.1)

                report_text = result.get("final_report", "")
                chunk_size = 80
                for i in range(0, len(report_text), chunk_size):
                    chunk = report_text[i : i + chunk_size]
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                    await asyncio.sleep(0.02)

                yield f"data: {json.dumps({'type': 'complete', 'citations': result.get('citations', []), 'sql_findings': result.get('sql_findings', {})})}\n\n"
            except Exception as ex:
                yield f"data: {json.dumps({'type': 'error', 'message': str(ex)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/marts/monthly")
    async def get_monthly_marts(company: Optional[str] = None):
        """Returns monthly sentiment time-series data."""
        try:
            marts_mgr.connect()
            df = marts_mgr.get_monthly_sentiment(company=company)
            marts_mgr.close()
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Failed to fetch monthly marts: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/marts/event_study")
    async def get_event_study(req: EventStudyRequest):
        """Calculates pre/post event study window around leadership events."""
        try:
            res = event_analyzer.compute_event_study(
                company=req.company,
                event_key=req.event_key,
                window_months=req.window_months,
            )
            return res
        except Exception as e:
            logger.error(f"Event study computation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/marts/did")
    async def get_did_analysis(req: DiDRequest):
        """Computes 2022-2026 Tech Layoffs Difference-in-Differences."""
        try:
            res = did_analyzer.compute_did(
                intervention_start=req.intervention_start,
                pre_window_months=req.pre_window_months,
                post_window_months=req.post_window_months,
            )
            return res
        except Exception as e:
            logger.error(f"DiD computation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


def run_server(host: str = "0.0.0.0", port: int = 8088):
    """Entrypoint to start the hosted server."""
    import uvicorn
    app = create_app()
    logger.info(f"Starting Reddit Morale Engine Agent Server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
