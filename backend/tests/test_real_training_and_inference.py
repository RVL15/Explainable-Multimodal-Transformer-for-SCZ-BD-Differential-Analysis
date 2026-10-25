import os
import json
import torch
import numpy as np
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.config import CHECKPOINTS_DIR, STANDARD_16_CHANNELS
from app.database import get_db, init_db
from app.datasets.ingestion import DatasetIngestionEngine
from app.models.pipeline import (
    get_subject_level_partitions,
    prepare_partition_dataset,
    preprocess_and_window_eeg
)
from app.eeg.io import load_eeg_file

client = TestClient(app)

def test_aseeg_directory_ingestion():
    """Verifies that DatasetIngestionEngine discovers and validates all 97 genuine ASEEG participants."""
    aseeg_dir = Path("backend/data/aseeg").resolve()
    if not (aseeg_dir / "EO").exists():
        pytest.skip("ASEEG EEG recordings directory not present.")

    summary = DatasetIngestionEngine.scan_and_ingest_directory(
        directory_path=str(aseeg_dir),
        dataset_id="aseeg",
        dataset_name="ASEEG Schizophrenia EEG Benchmark",
        target_task="SZ_VS_HC"
    )

    assert summary["is_ready_for_training"] is True
    assert summary["valid_participants_count"] == 97
    assert summary["class_distribution"] == {"HC": 50, "SZ": 47}
    assert summary["detected_channel_counts"] == [19]
    assert summary["detected_sampling_rates"] == [500.0]

def test_75_25_participant_split_zero_leakage():
    """Verifies strict 75% train/val and 25% test participant-level partition with ZERO leakage."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT subject_id, diagnostic_label, recording_path, is_synthetic
        FROM subjects
        WHERE dataset_id = 'aseeg' AND is_excluded = 0 AND file_exists = 1
        """)
        rows = cursor.fetchall()

    if len(rows) < 10:
        pytest.skip("ASEEG dataset not yet ingested in database.")

    subject_records = [dict(r) for r in rows]
    train_subs, val_subs, test_subs = get_subject_level_partitions(
        subject_records,
        test_ratio=0.25,
        val_ratio=0.15,
        random_seed=42
    )

    # 97 total: 25 test (ceil(97 * 0.25)), 72 train/val (75%)
    assert len(test_subs) == 25
    assert len(train_subs) + len(val_subs) == 72

    train_ids = {s["subject_id"] for s in train_subs}
    val_ids = {s["subject_id"] for s in val_subs}
    test_ids = {s["subject_id"] for s in test_subs}

    # Zero leakage assertion
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val subject overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test subject overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test subject overlap!"

def test_training_normalization_no_test_leakage():
    """Verifies that normalization parameters are fitted ONLY on training subjects."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM subjects WHERE dataset_id = 'aseeg' AND is_excluded = 0 LIMIT 10")
        subs = [dict(r) for r in cursor.fetchall()]

    if len(subs) < 6:
        pytest.skip("Insufficient subjects for test.")

    train_subs = subs[:4]
    test_subs = subs[4:6]

    # Fit normalization strictly on train
    X_train, y_train, _, train_norm_stats = prepare_partition_dataset(
        train_subs, target_task="SZ_VS_HC", norm_stats=None
    )
    assert train_norm_stats is not None
    assert train_norm_stats[0].shape == (1, 16, 1) # Channel-wise mean
    assert train_norm_stats[1].shape == (1, 16, 1) # Channel-wise std

    # Apply training stats to test without recalculating from test data
    X_test, y_test, _, applied_stats = prepare_partition_dataset(
        test_subs, target_task="SZ_VS_HC", norm_stats=train_norm_stats
    )
    assert np.allclose(applied_stats[0], train_norm_stats[0])
    assert np.allclose(applied_stats[1], train_norm_stats[1])

def test_saved_model_checkpoint_integrity():
    """Verifies that the best checkpoint exists, loads properly, and contains inference components."""
    best_cp = CHECKPOINTS_DIR / "best_model.pt"
    if not best_cp.exists():
        pytest.skip("No trained checkpoint found.")

    state = torch.load(str(best_cp), map_location="cpu")
    assert "model_state_dict" in state
    assert "norm_mean" in state
    assert "norm_std" in state
    assert "split_participant_ids" in state
    assert len(state["split_participant_ids"]["test"]) == 25
    assert "preprocessing_config" in state

def test_api_upload_eeg_prediction_flow():
    """Tests end-to-end EEG file upload and real model prediction."""
    sample_mat = Path("backend/data/aseeg/EO/sz/sz1.mat")
    if not sample_mat.exists():
        pytest.skip("Sample ASEEG recording not present.")

    with open(sample_mat, "rb") as f:
        response = client.post(
            "/api/multimodal/analyze",
            files={"eeg_file": ("sz1.mat", f, "application/octet-stream")},
            data={"uncertainty_threshold": "0.60"}
        )

    assert response.status_code == 200
    data = response.json()
    assert "predicted_label" in data
    assert data["predicted_label"] in ["SCZ", "BD", "SZ", "HC", "UNCERTAIN"]
    assert "prob_scz" in data
    assert "prob_bd" in data
    assert "confidence" in data
    assert "modality_evidence" in data
    assert "EEG" in data["modality_evidence"] or "eeg" in data["modality_evidence"]
    assert "eeg_channel_attributions" in data
    assert len(data["eeg_channel_attributions"]) == 16
