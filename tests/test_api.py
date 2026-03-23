from fastapi.testclient import TestClient
from app.main import app
from app.core.database import engine, Base
import pytest

# Initialize database for tests
Base.metadata.create_all(bind=engine)

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    payload = {
        "Time": 100.0,
        "Amount": 250.0,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "fraud_label" in data

def test_stats_endpoint():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_scanned" in data
    assert "fraud_detected" in data

def test_recent_transactions_endpoint():
    response = client.get("/recent-transactions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
