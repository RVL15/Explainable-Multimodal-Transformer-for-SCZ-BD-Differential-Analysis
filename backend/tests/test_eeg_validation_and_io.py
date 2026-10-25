import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from app.config import STANDARD_16_CHANNELS
from app.eeg.validation import validate_eeg_array, get_supported_formats_info
from app.eeg.io import load_eeg_file

def test_validation_passes_for_clean_eeg_array():
    # 16 channels, 1250 samples (5.0s at 250 Hz)
    rng = np.random.default_rng(42)
    clean_data = rng.standard_normal((16, 1250)) * 20.0 # ~20 uV amplitude
    chs = list(STANDARD_16_CHANNELS)
    srate = 250.0

    report = validate_eeg_array(clean_data, chs, srate)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0
    assert report["num_channels"] == 16
    assert report["duration_sec"] == 5.0
    assert report["quality_score"] >= 0.85

def test_validation_detects_nan_and_inf():
    rng = np.random.default_rng(42)
    bad_data = rng.standard_normal((16, 1000)) * 20.0
    bad_data[2, 100] = np.nan
    bad_data[5, 250] = np.inf

    report = validate_eeg_array(bad_data, list(STANDARD_16_CHANNELS), 250.0)
    assert report["is_valid"] is False
    assert any("NaN" in err for err in report["errors"])
    assert any("Infinite" in err for err in report["errors"])
    assert report["quality_score"] == 0.0

def test_validation_rejects_insufficient_duration():
    # Only 0.5 seconds at 250 Hz = 125 samples (minimum is 2.0s = 500 samples)
    short_data = np.random.randn(16, 125)
    report = validate_eeg_array(short_data, list(STANDARD_16_CHANNELS), 250.0, min_duration_sec=2.0)
    assert report["is_valid"] is False
    assert any("shorter than required minimum" in err for err in report["errors"])

def test_validation_rejects_invalid_sampling_rate():
    data = np.random.randn(16, 500)
    report_neg = validate_eeg_array(data, list(STANDARD_16_CHANNELS), -250.0)
    assert report_neg["is_valid"] is False
    assert any("non-positive" in err for err in report_neg["errors"])

    report_low = validate_eeg_array(data, list(STANDARD_16_CHANNELS), 20.0)
    assert report_low["is_valid"] is False
    assert any("below minimum Nyquist" in err for err in report_low["errors"])

def test_csv_eeg_file_loading():
    # Create temporary CSV EEG file: columns are channels, rows are time points
    n_times = 500
    df = pd.DataFrame({
        ch: np.random.randn(n_times) * 15.0 for ch in STANDARD_16_CHANNELS[:8]
    })
    
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_name = tmp.name
        df.to_csv(tmp_name, index=False)

    try:
        signals, channels, sample_rate = load_eeg_file(tmp_name)
        assert signals.shape == (8, 500)
        assert len(channels) == 8
        assert sample_rate == 250.0
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

def test_supported_formats_specification():
    info = get_supported_formats_info()
    assert ".npy" in info["supported_extensions"]
    assert ".edf" in info["supported_extensions"]
    assert ".fif" in info["supported_extensions"]
    assert ".csv" in info["supported_extensions"]
    assert "channel_count" in info["specifications"]
    assert "multimodal_inputs" in info

def test_detailed_eeg_validation_metadata_reporting():
    # 19 channels: 16 standard + 3 extra (Fz, Cz, Pz)
    channels_19 = list(STANDARD_16_CHANNELS) + ["Fz", "Cz", "Pz"]
    rng = np.random.default_rng(42)
    # 10.0 seconds at 500 Hz = 5000 samples
    data_500hz = rng.standard_normal((19, 5000)) * 15.0

    report = validate_eeg_array(
        data_500hz,
        channels_19,
        sample_rate=500.0,
        filename="subject_01_raw.mat",
        file_size_bytes=760000,
        file_format=".mat"
    )

    assert report["is_valid"] is True
    assert report["validation_status"] == "WARNINGS"  # Warning due to 500 Hz resampling and extra channels
    assert report["filename"] == "subject_01_raw.mat"
    assert report["file_format"] == ".mat"
    assert report["num_channels"] == 19
    assert report["num_standard_10_20_channels"] == 19
    assert len(report["selected_model_channels"]) == 16
    assert set(report["excluded_channels"]) == {"Fz", "Cz", "Pz"}
    assert len(report["missing_required_channels"]) == 0
    assert report["duration_sec"] == 10.0
    assert report["usable_samples"] == 5000
    assert report["total_windows_generated"] == 5
    assert report["usable_windows_count"] == 5
    assert report["has_nan_or_inf"] is False

    # Verify preprocessing status breakdown
    prep = report["preprocessing_status"]
    assert prep is not None
    assert prep["channel_alignment"]["status"] == "COMPLETED"
    assert "16/16 standard channels mapped" in prep["channel_alignment"]["details"]
    assert prep["resampling"]["status"] == "COMPLETED"
    assert "500.0 Hz to 250.0 Hz" in prep["resampling"]["details"]
    assert prep["bandpass_filtering"]["status"] == "CONFIGURED"
    assert prep["common_average_reference"]["status"] == "CONFIGURED"
    assert prep["artifact_rejection"]["status"] == "EVALUATED"
    assert prep["training_normalization"]["status"] == "CONFIGURED"

