from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from app.config import STANDARD_16_CHANNELS, DEFAULT_SAMPLE_RATE

SUPPORTED_EXTENSIONS = [".edf", ".bdf", ".fif", ".mat", ".npy", ".csv"]

def validate_eeg_array(
    signals: np.ndarray,
    channels: List[str],
    sample_rate: float,
    min_duration_sec: float = 2.0
) -> Dict[str, Any]:
    """
    Validates pre-loaded EEG data against neurophysiological and computational requirements:
    1. Dimension check (2D: channels x time).
    2. Finite value check (no NaNs, no Infs).
    3. Sampling rate validation (minimum 50 Hz, ideally 250 Hz).
    4. Duration check (must be at least min_duration_sec).
    5. Channel count and naming check.
    6. Signal dynamic range & flatline artifact check.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(signals, np.ndarray):
        errors.append("Signals data is not a valid NumPy ndarray.")
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "quality_score": 0.0
        }

    if signals.ndim != 2:
        errors.append(f"EEG signals must be 2-dimensional (channels x time). Received shape: {signals.shape}.")
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "quality_score": 0.0
        }

    num_channels, num_samples = signals.shape

    # 1. Non-empty check
    if num_channels == 0 or num_samples == 0:
        errors.append(f"EEG recording contains 0 channels or 0 samples. Shape: {signals.shape}.")
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "quality_score": 0.0
        }

    # 2. NaN and Inf check
    nan_count = int(np.isnan(signals).sum())
    inf_count = int(np.isinf(signals).sum())
    if nan_count > 0:
        errors.append(f"Recording contains {nan_count} NaN (Not a Number) values.")
    if inf_count > 0:
        errors.append(f"Recording contains {inf_count} Infinite (Inf) values.")

    # 3. Sampling rate check
    if sample_rate <= 0:
        errors.append(f"Invalid non-positive sampling rate: {sample_rate} Hz.")
    elif sample_rate < 50.0:
        errors.append(f"Sampling rate {sample_rate} Hz is below minimum Nyquist requirement (50 Hz).")
    elif abs(sample_rate - DEFAULT_SAMPLE_RATE) > 1e-2:
        warnings.append(
            f"Sampling rate is {sample_rate} Hz. Standard research model expects {DEFAULT_SAMPLE_RATE} Hz (resampling will be applied during preprocessing)."
        )

    # 4. Duration check
    duration_sec = num_samples / max(1.0, sample_rate)
    if duration_sec < min_duration_sec:
        errors.append(
            f"Recording duration ({duration_sec:.2f}s) is shorter than required minimum window ({min_duration_sec:.1f}s)."
        )

    # 5. Channel count check
    if num_channels < 4:
        warnings.append(f"Low channel count: {num_channels} channels. Standard 10-20 montage uses 16 channels.")
    elif num_channels < 16:
        warnings.append(f"Recording has {num_channels} channels; model will zero-pad to 16 channels.")

    # 6. Flatline & Amplitude checks
    std_per_channel = np.std(signals, axis=1) if num_samples > 1 else np.zeros(num_channels)
    flatline_channels = [channels[i] if i < len(channels) else f"Ch_{i+1}" for i in range(num_channels) if std_per_channel[i] < 1e-4]
    if flatline_channels:
        warnings.append(f"Flatline or disconnected electrode detected on: {', '.join(flatline_channels[:4])}.")

    max_val = float(np.max(np.abs(signals))) if num_samples > 0 and nan_count == 0 else 0.0
    if max_val < 0.01:
        warnings.append("Signal amplitudes appear very small (<0.01). If raw data is in Volts, conversion to microvolts (µV) is recommended.")
    elif max_val > 1500.0:
        warnings.append(f"Extreme peak voltage detected ({max_val:.1f} µV). Heavy ocular or muscular artifact present.")

    # Quality score heuristic
    is_valid = len(errors) == 0
    quality = 1.0
    if not is_valid:
        quality = 0.0
    else:
        quality -= 0.15 * len(warnings)
        quality = max(0.2, min(1.0, quality))

    # Channel match percentage
    matched_standard = [c for c in channels if c in STANDARD_16_CHANNELS]
    montage_compliance = f"{len(matched_standard)}/16 Standard 10-20 channels"

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "num_channels": num_channels,
        "num_samples": num_samples,
        "sample_rate": sample_rate,
        "duration_sec": round(duration_sec, 2),
        "channels": channels,
        "montage_compliance": montage_compliance,
        "quality_score": round(quality, 2),
        "peak_amplitude_uv": round(max_val, 2)
    }

def get_supported_formats_info() -> Dict[str, Any]:
    """
    Returns explicit specification of supported EEG formats and input constraints.
    """
    return {
        "supported_extensions": SUPPORTED_EXTENSIONS,
        "recommended_format": ".npy or .edf",
        "specifications": {
            "channel_count": "16 channels recommended (Standard 10-20 montage: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6)",
            "sampling_rate": "250 Hz (100 Hz - 1000 Hz supported with automatic resampling)",
            "window_duration": "2.0 seconds (500 samples per window at 250 Hz)",
            "filter_band": "1.0 Hz - 45.0 Hz zero-phase Butterworth bandpass filter",
            "reference": "Common Average Reference (CAR)",
            "artifact_threshold": "150.0 µV maximum amplitude / 300.0 µV peak-to-peak"
        },
        "multimodal_inputs": {
            "eeg": "16-channel time series recording (.npy, .edf, .fif, .mat, .csv)",
            "mri": "14 regional volumetric and cortical thickness ROI features (JSON or array)",
            "clinical": "8 neuropsychiatric items including PANSS, YMRS, MADRS, and demographics"
        }
    }
