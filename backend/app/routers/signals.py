import os
import tempfile
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from typing import Optional, List, Dict, Any
import numpy as np

from app.database import get_db
from app.eeg.io import load_eeg_file
from app.eeg.preprocessing import run_eeg_preprocessing
from app.eeg.spectral import compute_psd, compute_band_powers
from app.eeg.validation import validate_eeg_array, get_supported_formats_info
from app.schemas import PreprocessingOptions, EEGValidationResponse, SupportedFormatsResponse

router = APIRouter(prefix="/api/signals", tags=["Signals"])

def _find_subject_or_404(subject_id: str) -> Dict[str, Any]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM subjects WHERE subject_id = ?", (subject_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Subject {subject_id} not found in database.")
        sub = dict(row)
        if not sub.get("recording_path") or not os.path.exists(sub["recording_path"]):
            raise HTTPException(status_code=404, detail=f"Recording file for subject {subject_id} does not exist on disk.")
        return sub

@router.get("/supported-formats", response_model=SupportedFormatsResponse)
def get_supported_signal_formats():
    """
    Returns neurophysiological and computational specifications for supported input EEG formats.
    """
    return get_supported_formats_info()

@router.post("/validate", response_model=EEGValidationResponse)
async def validate_uploaded_eeg_file(
    file: UploadFile = File(...)
):
    """
    Validates an uploaded EEG file (.npy, .edf, .bdf, .fif, .mat, .csv):
    Verifies channel layout, sampling frequency, NaN/Inf presence, voltage scale, and duration.
    """
    suffix = Path(file.filename or "recording.npy").suffix.lower()
    if suffix not in [".npy", ".edf", ".bdf", ".fif", ".mat", ".csv"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Supported extensions: .npy, .edf, .bdf, .fif, .mat, .csv"
        )

    # Save to temp file for inspection
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        signals, channels, sample_rate = load_eeg_file(tmp_path)
        report = validate_eeg_array(
            signals,
            channels,
            sample_rate,
            filename=file.filename,
            file_size_bytes=len(content),
            file_format=suffix
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse and validate EEG file: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

@router.post("/upload-preview")
async def upload_and_preview_eeg(
    file: UploadFile = File(...),
    duration_sec: float = Query(5.0, ge=1.0, le=30.0),
    max_channels: int = Query(16, ge=1, le=64)
):
    """
    Uploads an EEG file, validates it, and returns waveform preview slices for interactive visualization.
    """
    suffix = Path(file.filename or "recording.npy").suffix.lower()
    if suffix not in [".npy", ".edf", ".bdf", ".fif", ".mat", ".csv"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Supported extensions: .npy, .edf, .bdf, .fif, .mat, .csv"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        signals, channels, sample_rate = load_eeg_file(tmp_path)
        val_report = validate_eeg_array(signals, channels, sample_rate)

        num_samples = min(int(duration_sec * sample_rate), signals.shape[1])
        snippet = signals[:max_channels, :num_samples]
        t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False).tolist()

        waveforms = []
        for ch_idx in range(snippet.shape[0]):
            ch_name = channels[ch_idx] if ch_idx < len(channels) else f"Ch_{ch_idx+1}"
            waveforms.append({
                "channel": ch_name,
                "raw_samples": [round(float(v), 3) for v in snippet[ch_idx]],
                "processed_samples": None
            })

        return {
            "filename": file.filename,
            "sample_rate": sample_rate,
            "duration_sec": round(num_samples / sample_rate, 2),
            "total_time_points": num_samples,
            "time_vector": [round(val, 4) for val in t],
            "channels": [w["channel"] for w in waveforms],
            "waveforms": waveforms,
            "validation_report": val_report
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to preview EEG file: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

@router.get("/{subject_id}/raw")
def get_raw_signal_waveform(
    subject_id: str,
    duration_sec: float = Query(5.0, ge=1.0, le=30.0),
    max_channels: int = Query(16, ge=1, le=64)
):
    """
    Returns time series snippet of raw EEG recording for interactive visualization.
    """
    sub = _find_subject_or_404(subject_id)
    signals, channels, sample_rate = load_eeg_file(sub["recording_path"])

    num_samples = min(int(duration_sec * sample_rate), signals.shape[1])
    snippet = signals[:max_channels, :num_samples]
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False).tolist()

    waveforms = []
    for ch_idx in range(snippet.shape[0]):
        ch_name = channels[ch_idx] if ch_idx < len(channels) else f"Ch_{ch_idx+1}"
        waveforms.append({
            "channel": ch_name,
            "raw_samples": [round(float(v), 3) for v in snippet[ch_idx]],
            "processed_samples": None
        })

    return {
        "subject_id": subject_id,
        "dataset_id": sub["dataset_id"],
        "diagnostic_label": sub["diagnostic_label"],
        "is_synthetic": bool(sub["is_synthetic"]),
        "sample_rate": sample_rate,
        "duration_sec": round(num_samples / sample_rate, 2),
        "total_time_points": num_samples,
        "time_vector": [round(val, 4) for val in t],
        "channels": [w["channel"] for w in waveforms],
        "waveforms": waveforms
    }

@router.post("/{subject_id}/preprocess")
def preprocess_eeg_signal(
    subject_id: str,
    options: PreprocessingOptions,
    duration_sec: float = Query(5.0, ge=1.0, le=30.0),
    max_channels: int = Query(16, ge=1, le=64)
):
    """
    Applies configurable 1-45 Hz bandpass filtering, CAR, channel normalization, and artifact rejection.
    Returns side-by-side raw vs processed waveforms and artifact rejection summary.
    """
    sub = _find_subject_or_404(subject_id)
    raw_signals, channels, sample_rate = load_eeg_file(sub["recording_path"])

    # Run preprocessing
    processed_signal, clean_windows, stats = run_eeg_preprocessing(
        raw_data=raw_signals,
        sample_rate=sample_rate,
        bandpass=options.bandpass_enabled,
        low_hz=options.low_freq_hz,
        high_hz=options.high_freq_hz,
        car=options.rereference_average,
        normalize=options.normalize_channels,
        window_sec=options.window_duration_sec,
        rejection_threshold_uv=options.rejection_threshold_uv
    )

    # Log preprocessing job in DB
    job_id = f"prep_{subject_id}_{int(options.low_freq_hz)}_{int(options.high_freq_hz)}"
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO preprocessing_jobs (
            id, subject_id, dataset_id, status, sample_rate, channels_count,
            duration_sec, total_windows, rejected_windows, rejection_ratio,
            filter_band, rereference
        ) VALUES (?, ?, ?, 'PROCESSED', ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            status = 'PROCESSED',
            total_windows = excluded.total_windows,
            rejected_windows = excluded.rejected_windows,
            rejection_ratio = excluded.rejection_ratio
        """, (
            job_id, subject_id, sub["dataset_id"], sample_rate, raw_signals.shape[0],
            float(raw_signals.shape[1] / sample_rate), stats["total_windows"], stats["rejected_windows"],
            stats["rejection_ratio"], f"{options.low_freq_hz}-{options.high_freq_hz} Hz",
            "CAR" if options.rereference_average else "None"
        ))

    # Slice for visualization
    num_samples = min(int(duration_sec * sample_rate), raw_signals.shape[1])
    raw_slice = raw_signals[:max_channels, :num_samples]
    proc_slice = processed_signal[:max_channels, :num_samples]
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False).tolist()

    waveforms = []
    for ch_idx in range(raw_slice.shape[0]):
        ch_name = channels[ch_idx] if ch_idx < len(channels) else f"Ch_{ch_idx+1}"
        waveforms.append({
            "channel": ch_name,
            "raw_samples": [round(float(v), 3) for v in raw_slice[ch_idx]],
            "processed_samples": [round(float(v), 3) for v in proc_slice[ch_idx]]
        })

    return {
        "subject_id": subject_id,
        "dataset_id": sub["dataset_id"],
        "diagnostic_label": sub["diagnostic_label"],
        "is_synthetic": bool(sub["is_synthetic"]),
        "sample_rate": sample_rate,
        "duration_sec": round(num_samples / sample_rate, 2),
        "total_time_points": num_samples,
        "time_vector": [round(val, 4) for val in t],
        "channels": [w["channel"] for w in waveforms],
        "waveforms": waveforms,
        "total_windows": stats["total_windows"],
        "rejected_windows": stats["rejected_windows"],
        "rejection_ratio": stats["rejection_ratio"],
        "applied_preprocessing": options.dict()
    }

@router.get("/{subject_id}/spectral")
def get_spectral_analysis(
    subject_id: str,
    max_channels: int = Query(8, ge=1, le=16)
):
    """
    Computes Power Spectral Density (PSD) and canonical frequency band breakdown.
    """
    sub = _find_subject_or_404(subject_id)
    raw_signals, channels, sample_rate = load_eeg_file(sub["recording_path"])

    freqs, psd = compute_psd(raw_signals, sample_rate)

    channels_spectral = []
    band_accum = {"Delta": 0.0, "Theta": 0.0, "Alpha": 0.0, "Beta": 0.0, "Gamma": 0.0}

    ch_limit = min(max_channels, raw_signals.shape[0])
    for ch_idx in range(ch_limit):
        ch_name = channels[ch_idx] if ch_idx < len(channels) else f"Ch_{ch_idx+1}"
        ch_psd = psd[ch_idx]
        bp_dict = compute_band_powers(freqs, ch_psd)

        band_items = []
        for b_name, b_data in bp_dict.items():
            band_items.append({
                "band": b_name,
                "freq_range": b_data["freq_range"],
                "absolute_power": b_data["absolute_power"],
                "relative_power": b_data["relative_power"]
            })
            band_accum[b_name] += b_data["relative_power"]

        # Downsample PSD for smooth rendering: pick ~40 frequency bins between 0.5 and 45 Hz
        step = max(1, len(freqs) // 40)
        channels_spectral.append({
            "channel": ch_name,
            "frequencies": [round(float(f), 2) for f in freqs[::step]],
            "psd_values": [round(float(p), 4) for p in ch_psd[::step]],
            "band_powers": band_items
        })

    mean_distribution = {k: round(v / ch_limit, 4) for k, v in band_accum.items()}

    return {
        "subject_id": subject_id,
        "channels_spectral": channels_spectral,
        "mean_band_distribution": mean_distribution
    }
