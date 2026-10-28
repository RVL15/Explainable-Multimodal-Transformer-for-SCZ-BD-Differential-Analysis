import os
import tempfile
import json
import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.main import app
from app.config import (
    CHECKPOINTS_DIR, STANDARD_16_CHANNELS, DEFAULT_SAMPLE_RATE,
    UNSEEN_DATA_DIR, UNSEEN_INCOMING_DIR, UNSEEN_RESULTS_DIR, UNSEEN_PROCESSED_DIR
)
from app.models.pipeline import run_unseen_eeg_inference


@pytest.fixture
def client():
    return TestClient(app)


def _generate_synthetic_clean_eeg(duration_sec=6.0, sample_rate=250.0, channels=None):
    if channels is None:
        channels = list(STANDARD_16_CHANNELS)
    n_samples = int(duration_sec * sample_rate)
    # Generate physiological plausible 10-30 uV sinusoidal oscillation
    t = np.linspace(0, duration_sec, n_samples)
    data = np.zeros((len(channels), n_samples), dtype=np.float64)
    for i in range(len(channels)):
        # Alpha (10 Hz) + Beta (20 Hz)
        data[i] = 15.0 * np.sin(2 * np.pi * 10 * t + i * 0.2) + 5.0 * np.sin(2 * np.pi * 20 * t)
    return data, channels, sample_rate


def test_unseen_directory_structure_exists():
    """Verify that backend/data/unseen/ structure exists with proper directories."""
    assert UNSEEN_DATA_DIR.exists(), "backend/data/unseen must exist"
    assert UNSEEN_INCOMING_DIR.exists(), "backend/data/unseen/incoming must exist"
    assert UNSEEN_PROCESSED_DIR.exists(), "backend/data/unseen/processed must exist"
    assert UNSEEN_RESULTS_DIR.exists(), "backend/data/unseen/results must exist"
    assert (UNSEEN_DATA_DIR / "README.md").exists(), "backend/data/unseen/README.md must exist"


def test_valid_unseen_eeg_npy_inference_and_metadata():
    """Verify valid unseen EEG inference loads best_model.pt and produces correct metadata."""
    sig, chs, srate = _generate_synthetic_clean_eeg()

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        np.save(tmp.name, sig)
        tmp_path = tmp.name

    try:
        res = run_unseen_eeg_inference(
            file_path=tmp_path,
            target_task="SZ_VS_HC",
            uncertainty_threshold=0.65,
            save_result=True
        )

        assert res["task"] == "SZ_VS_HC"
        assert res["model_checkpoint"] == "best_model.pt"
        assert res["is_new_input"] is True
        assert res["retrained_during_inference"] is False
        assert res["predicted_class"] in ["SZ", "HC", "UNCERTAIN"]
        assert "SZ" in res["class_scores"]
        assert "HC" in res["class_scores"]
        assert 0.0 <= res["class_scores"]["SZ"] <= 1.0
        assert 0.0 <= res["class_scores"]["HC"] <= 1.0
        assert res["usable_windows"] > 0
        assert res["explanation_available"] is True
        assert len(res["top_contributing_channels"]) > 0

        # Check that result JSON was written to backend/data/unseen/results/
        assert "saved_result_path" in res
        saved_file = Path(res["saved_result_path"])
        assert saved_file.exists()
        with open(saved_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
            assert saved_data["model_checkpoint"] == "best_model.pt"
            assert saved_data["task"] == "SZ_VS_HC"
            assert saved_data["retrained_during_inference"] is False

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_checkpoint_unmodified_during_inference():
    """Verify model checkpoint file is strictly read-only and hash does not change during inference."""
    best_ckpt = CHECKPOINTS_DIR / "best_model.pt"
    if not best_ckpt.exists():
        pytest.skip("best_model.pt not available")

    # Hash checkpoint before inference
    hasher_before = hashlib.sha256()
    with open(best_ckpt, "rb") as f:
        hasher_before.update(f.read())
    hash_before = hasher_before.hexdigest()
    mtime_before = best_ckpt.stat().st_mtime

    sig, chs, srate = _generate_synthetic_clean_eeg()
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        np.save(tmp.name, sig)
        tmp_path = tmp.name

    try:
        run_unseen_eeg_inference(tmp_path, target_task="SZ_VS_HC")

        hasher_after = hashlib.sha256()
        with open(best_ckpt, "rb") as f:
            hasher_after.update(f.read())
        hash_after = hasher_after.hexdigest()
        mtime_after = best_ckpt.stat().st_mtime

        assert hash_before == hash_after, "Checkpoint SHA256 was altered during inference! Must be immutable."
        assert mtime_before == mtime_after, "Checkpoint modification time changed during inference!"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_missing_channels_produce_clear_error():
    """Verify missing required channels trigger clear error and are not silently zero-filled."""
    # Only 14 channels (missing Fp1 and Fp2)
    incomplete_chs = STANDARD_16_CHANNELS[2:]
    sig, _, srate = _generate_synthetic_clean_eeg(channels=incomplete_chs)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        import pandas as pd
        df = pd.DataFrame(sig.T, columns=incomplete_chs)
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError) as excinfo:
            run_unseen_eeg_inference(tmp_path, target_task="SZ_VS_HC")
        assert "missing required" in str(excinfo.value).lower() or "channel" in str(excinfo.value).lower()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_unsupported_and_corrupted_files_rejected():
    """Verify unsupported extensions and corrupted files produce clear errors."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"not an eeg file")
        txt_path = tmp.name

    try:
        with pytest.raises(ValueError) as excinfo:
            run_unseen_eeg_inference(txt_path, target_task="SZ_VS_HC")
        assert "unsupported eeg file extension" in str(excinfo.value).lower()
    finally:
        if os.path.exists(txt_path):
            os.remove(txt_path)

    # Corrupted .mat file
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        tmp.write(b"CORRUPTED_BINARY_HEADER_DATA")
        mat_path = tmp.name

    try:
        with pytest.raises(Exception):
            run_unseen_eeg_inference(mat_path, target_task="SZ_VS_HC")
    finally:
        if os.path.exists(mat_path):
            os.remove(mat_path)


def test_scz_bd_rejected_when_checkpoint_absent():
    """Verify SCZ_VS_BD task is rejected with clinical privacy notice when scz_bd_model.pt is absent."""
    scz_bd_ckpt = CHECKPOINTS_DIR / "scz_bd_model.pt"
    if scz_bd_ckpt.exists():
        pytest.skip("scz_bd_model.pt exists in this environment")

    sig, chs, srate = _generate_synthetic_clean_eeg()
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        np.save(tmp.name, sig)
        tmp_path = tmp.name

    try:
        with pytest.raises(FileNotFoundError) as excinfo:
            run_unseen_eeg_inference(tmp_path, target_task="SCZ_VS_BD")
        err_msg = str(excinfo.value)
        assert "B-SNIP 1" in err_msg or "scz_bd_model.pt" in err_msg
        assert "Data Use Certification" in err_msg or "unavailable" in err_msg
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_api_unseen_incoming_and_predict_flow(client):
    """Verify REST API endpoints for unseen EEG workflow."""
    # 1. GET /api/signals/unseen/incoming
    r_inc = client.get("/api/signals/unseen/incoming")
    assert r_inc.status_code == 200
    inc_data = r_inc.json()
    assert "files" in inc_data
    assert "count" in inc_data

    # 2. GET /api/signals/unseen/results
    r_res = client.get("/api/signals/unseen/results")
    assert r_res.status_code == 200
    assert "results" in r_res.json()

    # 3. POST /api/signals/unseen/predict with direct upload
    sig, chs, srate = _generate_synthetic_clean_eeg()
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        np.save(tmp.name, sig)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            resp = client.post(
                "/api/signals/unseen/predict?target_task=SZ_VS_HC&uncertainty_threshold=0.65",
                files={"file": ("unseen_test_sample.npy", f, "application/octet-stream")}
            )
        assert resp.status_code == 200, f"API error: {resp.text}"
        data = resp.json()
        assert data["task"] == "SZ_VS_HC"
        assert data["model_checkpoint"] == "best_model.pt"
        assert data["retrained_during_inference"] is False
        assert data["predicted_class"] in ["SZ", "HC", "UNCERTAIN"]
        assert "class_scores" in data
        assert "saved_result_path" in data

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
