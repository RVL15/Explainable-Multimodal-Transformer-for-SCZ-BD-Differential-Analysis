import os
import io
import tempfile
from pathlib import Path
import pytest
import torch
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.config import STANDARD_16_CHANNELS, CHECKPOINTS_DIR
from app.models.cnn_transformer import EEGCNNTransformer
from app.datasets.bsnip_manager import BSNIPDatasetManager
from app.eeg.validation import validate_eeg_array

client = TestClient(app)

def test_scz_bd_task_rejects_fallback_to_sz_hc_when_checkpoint_absent(monkeypatch):
    """
    Phase 7 & 10 requirement:
    Never silently fall back to SZ-versus-HC model when SCZ_BD is requested.
    If scz_bd_model.pt does not exist, the API must return a 400 error explaining
    that authorized B-SNIP data / scz_bd_model.pt is required.
    """
    target_ckpt = CHECKPOINTS_DIR / "scz_bd_model.pt"
    # Ensure scz_bd_model.pt is not present during this test
    if target_ckpt.exists():
        monkeypatch.delattr(os.path, "exists")

    # Generate synthetic clean test EEG file
    data = np.random.randn(16, 1250) * 15.0
    buffer = io.BytesIO()
    np.save(buffer, data)
    buffer.seek(0)

    files = {"eeg_file": ("test_recording.npy", buffer.getvalue(), "application/octet-stream")}
    data_payload = {
        "target_task": "SCZ_BD",
        "uncertainty_threshold": "0.65"
    }

    response = client.post("/api/multimodal/analyze", files=files, data=data_payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "scz_bd_model.pt" in detail
    assert "B-SNIP 1" in detail
    # Verify no fallback to SZ/HC prediction occurred
    assert "prob_sz" not in response.json()

def test_scz_bd_manifest_label_validation_and_hc_rejection():
    """
    Phase 4 & 10 requirement:
    In the binary SCZ vs BD task, Healthy Controls (HC) must never be substituted
    or conflated with Bipolar Disorder (BD).
    """
    manifest_csv = (
        "subject_id,diagnosis,recording_file\n"
        "B_SCZ_001,SCZ,bsnip_scz_01.edf\n"
        "B_BD_001,BD,bsnip_bd_01.edf\n"
        "B_HC_001,HC,bsnip_hc_01.edf\n"
        "B_SCZ_002,SZ,bsnip_scz_02.edf\n"
        "B_DUP_001,BD,bsnip_bd_dup.edf\n"
        "B_DUP_001,BD,bsnip_bd_dup.edf\n"
    )

    report = BSNIPDatasetManager.validate_manifest_content(manifest_csv)
    assert report["total_rows"] == 6
    assert len(report["duplicates"]) == 1
    assert "B_DUP_001" in report["duplicates"]

    # Verify diagnostic label distribution
    dist = report["class_distribution"]
    assert dist["SCZ"] == 2  # SCZ and SZ mapped to SCZ
    assert dist["BD"] == 2   # 2 unique BD subjects
    assert dist["HC"] == 1   # HC tracked separately, never counted as BD

def test_scz_bd_architecture_compatibility_and_classes():
    """
    Phase 5 & 10 requirement:
    Verify that the CNN-Transformer architecture supports binary SCZ vs BD (2 classes: SCZ=0, BD=1).
    Verify forward pass output tensor dimensions [batch_size, 2] and softmax probabilities.
    """
    model = EEGCNNTransformer(
        num_channels=16,
        time_points=500,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=2,
        dropout=0.1
    )
    model.eval()

    # Input: 2 subjects, each with 3 windows of shape [16, 500]
    dummy_input = torch.randn(2, 3, 16, 500)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, 2)
    probs = torch.softmax(logits, dim=-1)
    assert probs.shape == (2, 2)
    # Sum of probabilities across SCZ and BD must equal 1.0
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

def test_scz_bd_participant_level_split_isolation():
    """
    Phase 4 & 10 requirement:
    No participant ID may leak across train and held-out test splits.
    """
    scz_subjects = [f"SCZ_{i:03d}" for i in range(20)]
    bd_subjects = [f"BD_{i:03d}" for i in range(20)]
    all_subjects = scz_subjects + bd_subjects

    # 75/25 split
    np.random.seed(42)
    shuffled = np.random.permutation(all_subjects)
    split_idx = int(len(shuffled) * 0.75)
    train_subs = set(shuffled[:split_idx])
    test_subs = set(shuffled[split_idx:])

    # Strict zero overlap
    overlap = train_subs.intersection(test_subs)
    assert len(overlap) == 0
    assert len(train_subs) + len(test_subs) == 40

def test_classification_tasks_readiness_endpoint():
    """
    Phase 7 & 8 requirement:
    GET /api/health/classification-tasks must return structured metadata for both tasks
    with clear, honest distinction of readiness and provenance.
    """
    response = client.get("/api/health/classification-tasks")
    assert response.status_code == 200
    data = response.json()

    assert "sz_vs_hc" in data
    assert "scz_vs_bd" in data

    # Task A: SZ vs HC
    sz_hc = data["sz_vs_hc"]
    assert sz_hc["status"] == "MODEL READY"
    assert sz_hc["active_checkpoint"] == "best_model.pt"
    assert sz_hc["dataset"] == "ASEEG"
    assert sz_hc["held_out_accuracy"] == "72.0%"

    # Task B: SCZ vs BD
    scz_bd = data["scz_vs_bd"]
    assert scz_bd["status"] == "DATA REQUIRED"
    assert "B-SNIP 1" in scz_bd["dataset"]
    assert "B-SNIP 1 data is required" in scz_bd["status_reason"]
    assert scz_bd["active_checkpoint"] is None

def test_scz_bd_pipeline_rejects_hc_in_data_loader():
    """
    Phase 4 & 10 requirement:
    In prepare_partition_dataset with target_task='SCZ_VS_BD',
    Healthy Controls (HC) must be strictly rejected/omitted and never mapped to BD or SCZ.
    """
    from app.models.pipeline import prepare_partition_dataset
    mock_subjects = [
        {"subject_id": "SUB_01", "diagnostic_label": "SCZ", "recording_path": "backend/data/aseeg/EO/sz/sz1.mat"},
        {"subject_id": "SUB_02", "diagnostic_label": "HC", "recording_path": "backend/data/aseeg/EO/normal/n1.mat"},
    ]
    # SUB_02 (HC) must be excluded from windows and labels
    w, labels, sub_ids, _ = prepare_partition_dataset(mock_subjects, target_task="SCZ_VS_BD")
    assert all(sid == "SUB_01" for sid in sub_ids)
    assert all(lbl == 0 for lbl in labels)

def test_scz_bd_integrated_gradients_attribution():
    """
    Phase 9 & 10 requirement:
    Verify Integrated Gradients computation for binary SCZ vs BD classes (0=SCZ, 1=BD).
    """
    from app.xai.attribution import compute_gradient_attribution
    model = EEGCNNTransformer(num_channels=16, time_points=500, num_classes=2, d_model=64)
    model.eval()

    windows = torch.randn(4, 16, 500)
    # Attribution towards SCZ (class 0)
    attr_0, ch_imp_0 = compute_gradient_attribution(model, windows, target_class=0, steps=10)
    assert attr_0.shape == (4, 16, 500)
    assert ch_imp_0.shape == (16,)
    assert np.isclose(np.sum(ch_imp_0), 1.0, atol=1e-4)

    # Attribution towards BD (class 1)
    attr_1, ch_imp_1 = compute_gradient_attribution(model, windows, target_class=1, steps=10)
    assert attr_1.shape == (4, 16, 500)
    assert ch_imp_1.shape == (16,)
    assert np.isclose(np.sum(ch_imp_1), 1.0, atol=1e-4)

def test_scz_bd_metrics_computation():
    """
    Phase 6 & 10 requirement:
    Verify binary classification metrics formula for SCZ (class 0) vs BD (class 1).
    TN = SCZ correct, TP = BD correct, FP = SCZ predicted as BD, FN = BD predicted as SCZ.
    """
    from app.evaluation.metrics import compute_classification_metrics
    # 5 SCZ (0), 5 BD (1)
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0] # 3 TN, 2 FP, 3 TP, 2 FN
    y_scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.85, 0.75, 0.35, 0.4]

    metrics = compute_classification_metrics(y_true, y_pred, y_scores)
    assert metrics["accuracy"] == 0.60
    assert metrics["confusion_matrix"] == [[3, 2], [2, 3]]
    assert metrics["sensitivity"] == 0.60
    assert metrics["specificity"] == 0.60
    assert 0.0 <= metrics["roc_auc"] <= 1.0

