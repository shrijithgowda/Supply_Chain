import pytest
from fastapi.testclient import TestClient
from api.main import app
import os

client = TestClient(app)
API_KEY = os.getenv("API_KEY", "prod_secret_key_123")

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_forecast_unauthorized():
    response = client.post("/forecast", json={
        "item_id": "FOODS_3_090",
        "store_id": "CA_1"
    })
    assert response.status_code == 403

def test_forecast_authorized():
    response = client.post(
        "/forecast",
        json={"item_id": "FOODS_3_090", "store_id": "CA_1"},
        headers={"x-api-key": API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert "point_forecast" in data
    assert len(data["point_forecast"]) == 28

def test_invalid_input():
    response = client.post(
        "/forecast",
        json={"item_id": "FOODS_3_090"}, # Missing store_id
        headers={"x-api-key": API_KEY}
    )
    assert response.status_code == 422
