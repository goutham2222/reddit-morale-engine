"""Unit tests for Microsoft Foundry Hosted Agent Server & FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.agents.server import create_app
from src.common.config import EngineConfig


@pytest.fixture
def client():
    config = EngineConfig.load()
    app = create_app(config)
    return TestClient(app)


def test_server_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["port"] == 8088


def test_server_info(client):
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "Reddit Morale Engine" in data["name"]
    assert "LangGraph" in data["architecture"]


def test_server_research_endpoint(client):
    payload = {
        "query": "How did Andy Jassy's 5-day RTO announcement impact Amazon employee morale in 2024?",
        "companies": ["Amazon"],
        "top_k_evidence": 5,
    }
    response = client.post("/api/research", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "final_report" in data
    assert len(data["final_report"]) > 0
    assert "target_companies" in data


def test_server_research_stream_endpoint(client):
    payload = {
        "query": "What was the employee reaction to Meta's Year of Efficiency?",
        "companies": ["Meta"],
        "top_k_evidence": 3,
    }
    response = client.post("/api/research/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    text = response.text
    assert "data:" in text
