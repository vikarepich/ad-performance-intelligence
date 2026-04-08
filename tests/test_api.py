"""
Tests for the FastAPI Backend (src/api/main.py).

What's happening here:
- We test all API endpoints using FastAPI's TestClient
- TestClient simulates HTTP requests without running a real server
- We test both successful responses and error handling

New concept — TestClient:
FastAPI provides a TestClient that lets you call endpoints
in tests as if you were making real HTTP requests.
No need to start uvicorn — it all runs in-process.

    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.main import app


# ============================================================
# FIXTURE
# ============================================================

@pytest.fixture
def client():
    """
    Create a FastAPI TestClient.

    TestClient wraps the app and lets us call endpoints
    with .get(), .post() just like the requests library.
    """
    return TestClient(app)


# ============================================================
# TESTS FOR /health
# ============================================================

class TestHealth:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, client):
        """Check that /health returns status code 200."""
        response = client.get("/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    def test_health_returns_ok(self, client):
        """Check that /health returns {"status": "ok"}."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok", f"Expected 'ok', got {data['status']}"


# ============================================================
# TESTS FOR /campaigns
# ============================================================

class TestCampaigns:
    """Tests for the campaigns endpoint."""

    def test_campaigns_returns_200(self, client):
        """Check that /campaigns returns 200."""
        response = client.get("/campaigns")
        assert response.status_code == 200

    def test_campaigns_has_count(self, client):
        """Check that response has a count field."""
        response = client.get("/campaigns")
        data = response.json()
        assert "count" in data, "Response should have 'count'"
        assert "campaigns" in data, "Response should have 'campaigns'"

    def test_campaigns_count_matches_list(self, client):
        """Check that count matches the number of campaigns returned."""
        response = client.get("/campaigns")
        data = response.json()
        assert data["count"] == len(data["campaigns"]), \
            "Count should match campaigns list length"

    def test_campaigns_filter_by_name(self, client):
        """Check that campaign_name filter works."""
        all_response = client.get("/campaigns")
        all_data = all_response.json()

        if all_data["count"] > 0:
            name = all_data["campaigns"][0]["campaign_name"]
            filtered = client.get(f"/campaigns?campaign_name={name}")
            filtered_data = filtered.json()

            for campaign in filtered_data["campaigns"]:
                assert campaign["campaign_name"] == name, \
                    f"Expected campaign '{name}', got '{campaign['campaign_name']}'"

    def test_campaigns_anomalies_only(self, client):
        """Check that anomalies_only filter works."""
        response = client.get("/campaigns?anomalies_only=true")
        data = response.json()

        for campaign in data["campaigns"]:
            assert campaign["is_anomaly"] == 1, \
                "All campaigns should be anomalies when filtered"


# ============================================================
# TESTS FOR /anomalies
# ============================================================

class TestAnomalies:
    """Tests for the anomalies endpoint."""

    def test_anomalies_returns_200(self, client):
        """Check that /anomalies returns 200."""
        response = client.get("/anomalies")
        assert response.status_code == 200

    def test_anomalies_has_required_fields(self, client):
        """Check that response has count, total_rows, anomaly_rate."""
        response = client.get("/anomalies")
        data = response.json()

        assert "count" in data, "Should have 'count'"
        assert "total_rows" in data, "Should have 'total_rows'"
        assert "anomaly_rate" in data, "Should have 'anomaly_rate'"
        assert "anomalies" in data, "Should have 'anomalies'"

    def test_anomaly_rate_is_percentage(self, client):
        """Check that anomaly_rate is between 0 and 100."""
        response = client.get("/anomalies")
        data = response.json()

        assert 0 <= data["anomaly_rate"] <= 100, \
            f"Anomaly rate should be 0-100%, got {data['anomaly_rate']}"


# ============================================================
# TESTS FOR /metrics
# ============================================================

class TestMetrics:
    """Tests for the metrics endpoint."""

    def test_metrics_returns_200(self, client):
        """Check that /metrics returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_has_models(self, client):
        """Check that metrics include anomaly and/or forecaster."""
        response = client.get("/metrics")
        data = response.json()

        has_anomaly = "anomaly_detector" in data
        has_forecaster = "roas_forecaster" in data

        assert has_anomaly or has_forecaster, \
            "Should have at least one model's metrics"


# ============================================================
# TESTS FOR /importance
# ============================================================

class TestImportance:
    """Tests for the feature importance endpoint."""

    def test_importance_returns_200(self, client):
        """Check that /importance returns 200."""
        response = client.get("/importance")
        assert response.status_code == 200

    def test_importance_has_features(self, client):
        """Check that importance data has feature names and values."""
        response = client.get("/importance")
        data = response.json()

        for model_name in data:
            importance_list = data[model_name]
            assert len(importance_list) > 0, f"{model_name} should have features"

            for item in importance_list:
                assert "feature" in item, "Each item should have 'feature'"
                assert "importance" in item, "Each item should have 'importance'"
                assert item["importance"] >= 0, "Importance should be non-negative"


# ============================================================
# TESTS FOR /summary
# ============================================================

class TestSummary:
    """Tests for the summary endpoint."""

    def test_summary_returns_200(self, client):
        """Check that /summary returns 200."""
        response = client.get("/summary")
        assert response.status_code == 200

    def test_summary_has_required_fields(self, client):
        """Check that summary has all expected fields."""
        response = client.get("/summary")
        data = response.json()

        assert "total_rows" in data, "Should have 'total_rows'"
        assert "unique_campaigns" in data, "Should have 'unique_campaigns'"
        assert "date_range" in data, "Should have 'date_range'"
        assert "average_metrics" in data, "Should have 'average_metrics'"
        assert "anomalies" in data, "Should have 'anomalies'"
        assert "best_campaign" in data, "Should have 'best_campaign'"
        assert "worst_campaign" in data, "Should have 'worst_campaign'"

    def test_summary_best_has_higher_roas_than_worst(self, client):
        """Check that best campaign ROAS > worst campaign ROAS."""
        response = client.get("/summary")
        data = response.json()

        best_roas = data["best_campaign"]["avg_roas"]
        worst_roas = data["worst_campaign"]["avg_roas"]

        assert best_roas >= worst_roas, \
            f"Best ROAS ({best_roas}) should be >= worst ({worst_roas})"


# ============================================================
# TESTS FOR /predict
# ============================================================

class TestPredict:
    """Tests for the predict endpoint."""

    def test_predict_returns_200(self, client):
        """Check that /predict returns 200 with valid input."""
        payload = {
            "ctr": 0.04,
            "cpc": 2.5,
            "cpl": 12.0,
            "conversion_rate": 0.10,
            "ctr_wow": -0.20,
            "spend_wow": 0.15,
            "ctr_rolling3": 0.05,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_has_roas_forecast(self, client):
        """Check that prediction includes ROAS forecast."""
        payload = {
            "ctr": 0.04,
            "cpc": 2.5,
            "cpl": 12.0,
            "conversion_rate": 0.10,
            "ctr_wow": -0.20,
            "spend_wow": 0.15,
            "ctr_rolling3": 0.05,
        }
        response = client.post("/predict", json=payload)
        data = response.json()

        assert "roas_forecast" in data, "Should have 'roas_forecast'"

    def test_predict_with_anomaly_data(self, client):
        """Check that full payload returns both forecasts and anomaly detection."""
        payload = {
            "ctr": 0.04,
            "cpc": 2.5,
            "cpl": 12.0,
            "conversion_rate": 0.10,
            "ctr_wow": -0.20,
            "spend_wow": 0.15,
            "ctr_rolling3": 0.05,
            "roas": 3.0,
            "roas_wow": -0.30,
            "roas_rolling3": 3.5,
        }
        response = client.post("/predict", json=payload)
        data = response.json()

        assert "roas_forecast" in data, "Should have 'roas_forecast'"
        assert "anomaly_detection" in data, "Should have 'anomaly_detection'"

    def test_predict_validates_input(self, client):
        """
        Check that invalid input returns 422 (validation error).

        FastAPI + Pydantic automatically validate input.
        Missing required fields or wrong types trigger 422.
        """
        payload = {"ctr": 0.04}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422, \
            "Missing required fields should return 422"


# ============================================================
# TESTS FOR /ask
# ============================================================

class TestAsk:
    """
    Tests for the RAG ask endpoint.

    We mock the RAG pipeline to avoid calling the real LLM
    and ChromaDB during tests.
    """

    @patch("src.api.main.rag_ask", create=True)
    def test_ask_returns_200(self, mock_rag, client):
        """Check that /ask returns 200 with valid input."""
        with patch("src.rag.chain.ask") as mock_ask:
            mock_ask.return_value = {
                "answer": "Banner partner has the worst ROAS.",
                "sources": ["doc1", "doc2"],
                "metadatas": [{"campaign_name": "banner_partner"}],
                "distances": [0.5],
            }

            payload = {"question": "Which campaign is worst?"}
            response = client.post("/ask", json=payload)
            assert response.status_code == 200

    def test_ask_validates_input(self, client):
        """Check that missing question returns 422."""
        payload = {}
        response = client.post("/ask", json=payload)
        assert response.status_code == 422, \
            "Missing question should return 422"