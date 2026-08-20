import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Only run API tests if artifacts exist
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACTS_EXIST = (ARTIFACTS_DIR / "model_mean.joblib").exists()


@pytest.fixture(scope="module")
def client():
    from app.main import app
    from app.models.predictor import Predictor

    # Manually load predictor since TestClient may not trigger lifespan
    predictor = Predictor()
    predictor.load()
    app.state.predictor = predictor

    with TestClient(app) as c:
        yield c


@pytest.mark.skipif(not ARTIFACTS_EXIST, reason="Model artifacts not generated yet")
class TestAPI:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_landing_page(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "DecisionIQ" in resp.text

    def test_results_page(self, client):
        resp = client.get("/results")
        assert resp.status_code == 200

    def test_predict_page(self, client):
        resp = client.get("/predict")
        assert resp.status_code == 200

    def test_simulation_page(self, client):
        resp = client.get("/simulation")
        assert resp.status_code == 200

    def test_scatter_data_api(self, client):
        resp = client.get("/api/results/scatter-data")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        assert "actual" in data[0]

    def test_interval_data_api(self, client):
        resp = client.get("/api/results/interval-data")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 50

    def test_results_table_api(self, client):
        resp = client.get("/api/results/table?page=1")
        assert resp.status_code == 200

    def test_predict_api(self, client):
        resp = client.post("/api/predict", data={
            "invoice_amount": 50000,
            "posting_date": "2020-06-15",
            "payment_terms": "NAA8",
            "business_code": "U001",
            "cust_number": "",
        })
        assert resp.status_code == 200

    def test_random_invoice_api(self, client):
        resp = client.get("/api/predict/random")
        assert resp.status_code == 200
        data = resp.json()
        assert "invoice_amount" in data

    def test_simulate_api(self, client):
        resp = client.post("/api/simulate", data={
            "capacity": 10,
            "cost_per_call": 15,
            "capital_cost_rate": 0.0003,
            "days_accelerated": 3,
            "batch_size": 50,
            "n_days": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
