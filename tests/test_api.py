"""
Unit tests for the day-ahead electricity load forecaster API.
Verifies FastAPI endpoints using TestClient.
"""

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_read_health():
    """
    Verifies that the /health diagnostic endpoint works.
    """
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert "model_loaded" in json_data


def test_model_info_error_if_missing():
    """
    Verifies /model-info returns 404 or 200 properly.
    """
    response = client.get("/model-info")
    # If the pipeline hasn't run yet, it could return 404, else 200
    assert response.status_code in [200, 404]
