import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_root_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "Explainable Multimodal Transformer" in data["project"]

def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "models_status" in data
    assert "checkpoints" in data["models_status"]
    assert "supported_input_formats" in data
    assert ".npy" in data["supported_input_formats"]

def test_supported_formats_endpoint(client):
    res = client.get("/api/signals/supported-formats")
    assert res.status_code == 200
    data = res.json()
    assert ".edf" in data["supported_extensions"]
    assert "multimodal_inputs" in data

def test_multimodal_features_schema_endpoint(client):
    res = client.get("/api/multimodal/features-schema")
    assert res.status_code == 200
    data = res.json()
    assert "mri_roi_features" in data
    assert "clinical_features" in data
    assert len(data["mri_roi_features"]) == 14
    assert len(data["clinical_features"]) == 8

def test_multimodal_analyze_with_cohort_subject(client):
    # Pick first demo subject
    subs_res = client.get("/api/xai/subjects?dataset_id=demo")
    assert subs_res.status_code == 200
    subs = subs_res.json()
    assert len(subs) > 0
    first_sub = subs[0]["subject_id"]

    res = client.post(
        "/api/multimodal/analyze",
        data={
            "subject_id": first_sub,
            "uncertainty_threshold": 0.65
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert "predicted_label" in data
    assert data["predicted_label"] in ["SCZ", "BD", "UNCERTAIN"]
    assert "prob_scz" in data
    assert "prob_bd" in data
    assert "modality_evidence" in data
    assert "modality_agreement" in data
    assert "scientific_disclaimer" in data

def test_demo_cohort_status(client):
    res = client.get("/api/demo/status")
    assert res.status_code == 200
    data = res.json()
    assert "synthetic_subjects_count" in data
    assert data["marker"] == "[SYNTHETIC DEMO]"
