import os
import csv
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

from app.config import STANDARD_16_CHANNELS, DEFAULT_SAMPLE_RATE

def load_eeg_file(file_path: str) -> Tuple[np.ndarray, List[str], float]:
    """
    Loads EEG recording from EDF, EDF+, FIF, MAT, NPY, or CSV files.
    Returns:
        signals: np.ndarray of shape (num_channels, num_samples)
        channels: list of channel names
        sample_rate: sampling frequency in Hz
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"EEG file not found: {file_path}")

    ext = p.suffix.lower()

    if ext in [".edf", ".bdf"]:
        try:
            import mne
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            channels = list(raw.ch_names)
            sample_rate = float(raw.info["sfreq"])
            signals = raw.get_data() # shape (n_channels, n_times) in Volts
            # Convert to microvolts if in Volts
            if np.max(np.abs(signals)) < 0.01:
                signals = signals * 1e6
            return signals, channels, sample_rate
        except Exception as e:
            raise RuntimeError(f"Failed to read EDF file via MNE: {str(e)}")

    elif ext == ".fif":
        try:
            import mne
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
            channels = list(raw.ch_names)
            sample_rate = float(raw.info["sfreq"])
            signals = raw.get_data()
            if np.max(np.abs(signals)) < 0.01:
                signals = signals * 1e6
            return signals, channels, sample_rate
        except Exception as e:
            raise RuntimeError(f"Failed to read FIF file: {str(e)}")

    elif ext == ".mat":
        try:
            arr = None
            sample_rate = float(DEFAULT_SAMPLE_RATE)
            try:
                from scipy.io import loadmat
                mat = loadmat(file_path)
                for k in ["preprocessed_data", "data", "eeg", "val", "signal", "recordings"]:
                    if k in mat and isinstance(mat[k], np.ndarray):
                        arr = mat[k]
                        break
                if arr is None:
                    for k, v in mat.items():
                        if isinstance(v, np.ndarray) and v.ndim == 2:
                            arr = v
                            break
                if "fs" in mat:
                    sample_rate = float(mat["fs"])
            except NotImplementedError:
                # MATLAB v7.3 HDF5 format
                import h5py
                with h5py.File(file_path, "r") as f:
                    for k in ["preprocessed_data", "data", "eeg", "val", "signal", "recordings"]:
                        if k in f:
                            arr = np.array(f[k], dtype=np.float64)
                            break
                    if arr is None:
                        for k in f.keys():
                            ds = f[k]
                            if hasattr(ds, "ndim") and ds.ndim == 2:
                                arr = np.array(ds, dtype=np.float64)
                                break
                    if "fs" in f:
                        sample_rate = float(np.array(f["fs"]).squeeze())
            
            if arr is None:
                raise ValueError("No 2D signal matrix found in MATLAB file.")

            # Ensure shape is (n_channels, n_times)
            if arr.shape[0] > arr.shape[1]:
                arr = arr.T
            
            # Map standard channels
            if arr.shape[0] == 16:
                channels = list(STANDARD_16_CHANNELS)
            elif arr.shape[0] == 19:
                channels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
                sample_rate = 500.0  # Standard ASEEG 500 Hz acquisition rate
            else:
                channels = [f"Ch{i+1}" for i in range(arr.shape[0])]
            
            return arr.astype(np.float64), channels, sample_rate
        except Exception as e:
            raise RuntimeError(f"Failed to read MAT file: {str(e)}")

    elif ext == ".npy":
        arr = np.load(file_path)
        if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
            arr = arr.T
        if arr.shape[0] == 16:
            channels = list(STANDARD_16_CHANNELS)
        else:
            channels = [f"Ch{i+1}" for i in range(arr.shape[0])]
        sample_rate = float(DEFAULT_SAMPLE_RATE)
        return arr.astype(np.float64), channels, sample_rate

    elif ext == ".csv":
        try:
            df = pd.read_csv(file_path)
            # Check if columns are channels or time points
            if df.shape[1] <= 128 and df.shape[0] > df.shape[1]:
                # Standard tabular EEG: columns = channels, rows = time points
                channels = list(df.columns)
                arr = df.to_numpy(dtype=np.float64).T # Transpose to (channels, time)
            else:
                # rows = channels, cols = time points
                channels = [f"Ch{i+1}" for i in range(df.shape[0])]
                arr = df.to_numpy(dtype=np.float64)
            sample_rate = float(DEFAULT_SAMPLE_RATE)
            return arr, channels, sample_rate
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV EEG file: {str(e)}")

    else:
        raise ValueError(f"Unsupported EEG file extension: {ext}. Supported: .edf, .bdf, .fif, .mat, .npy, .csv")

def get_eeg_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extracts EEG file properties (channels, sample rate, duration, size)
    without loading the full signal array into RAM.
    Essential for chunked/streaming ingestion of large multi-gigabyte datasets.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"EEG file not found: {file_path}")

    ext = p.suffix.lower()
    file_size = p.stat().st_size

    if ext in [".edf", ".bdf"]:
        try:
            import mne
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
            channels = list(raw.ch_names)
            srate = float(raw.info["sfreq"])
            n_samples = int(raw.n_times)
            duration = float(n_samples / srate) if srate > 0 else 0.0
            return {
                "file_path": str(p),
                "file_size_bytes": file_size,
                "file_format": ext,
                "sample_rate": srate,
                "channels_count": len(channels),
                "channels": channels,
                "duration_sec": round(duration, 2),
                "n_samples": n_samples
            }
        except Exception as e:
            raise RuntimeError(f"EDF metadata read error: {str(e)}")

    elif ext == ".fif":
        try:
            import mne
            raw = mne.io.read_raw_fif(file_path, preload=False, verbose=False)
            channels = list(raw.ch_names)
            srate = float(raw.info["sfreq"])
            n_samples = int(raw.n_times)
            duration = float(n_samples / srate) if srate > 0 else 0.0
            return {
                "file_path": str(p),
                "file_size_bytes": file_size,
                "file_format": ext,
                "sample_rate": srate,
                "channels_count": len(channels),
                "channels": channels,
                "duration_sec": round(duration, 2),
                "n_samples": n_samples
            }
        except Exception as e:
            raise RuntimeError(f"FIF metadata read error: {str(e)}")

    elif ext == ".mat":
        try:
            # First attempt HDF5 (MATLAB v7.3) without reading data array
            import h5py
            if h5py.is_hdf5(file_path):
                with h5py.File(file_path, "r") as f:
                    arr_shape = None
                    for k in ["preprocessed_data", "data", "eeg", "val", "signal", "recordings"]:
                        if k in f and hasattr(f[k], "shape"):
                            arr_shape = f[k].shape
                            break
                    if arr_shape is None:
                        for k in f.keys():
                            if hasattr(f[k], "shape") and len(f[k].shape) == 2:
                                arr_shape = f[k].shape
                                break
                    if arr_shape:
                        n_ch = min(arr_shape[0], arr_shape[1])
                        n_samp = max(arr_shape[0], arr_shape[1])
                    else:
                        n_ch, n_samp = 16, 0

                    srate = 500.0 if n_ch == 19 else float(DEFAULT_SAMPLE_RATE)
                    if "fs" in f:
                        srate = float(np.array(f["fs"]).squeeze())

                    chs = list(STANDARD_16_CHANNELS) if n_ch == 16 else (
                        ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
                        if n_ch == 19 else [f"Ch{i+1}" for i in range(n_ch)]
                    )
                    dur = round(float(n_samp / srate), 2) if srate > 0 else 0.0
                    return {
                        "file_path": str(p),
                        "file_size_bytes": file_size,
                        "file_format": ext,
                        "sample_rate": srate,
                        "channels_count": n_ch,
                        "channels": chs,
                        "duration_sec": dur,
                        "n_samples": n_samp
                    }
        except Exception:
            pass

        # Fallback to load_eeg_file
        signals, chs, srate = load_eeg_file(file_path)
        dur = round(float(signals.shape[1] / srate), 2) if srate > 0 else 0.0
        return {
            "file_path": str(p),
            "file_size_bytes": file_size,
            "file_format": ext,
            "sample_rate": srate,
            "channels_count": len(chs),
            "channels": chs,
            "duration_sec": dur,
            "n_samples": signals.shape[1]
        }

    elif ext == ".npy":
        # Memory-mapped NPY inspection
        mmap = np.load(file_path, mmap_mode="r")
        n_ch = min(mmap.shape[0], mmap.shape[1]) if mmap.ndim == 2 else mmap.shape[0]
        n_samp = max(mmap.shape[0], mmap.shape[1]) if mmap.ndim == 2 else 0
        srate = float(DEFAULT_SAMPLE_RATE)
        chs = list(STANDARD_16_CHANNELS) if n_ch == 16 else [f"Ch{i+1}" for i in range(n_ch)]
        return {
            "file_path": str(p),
            "file_size_bytes": file_size,
            "file_format": ext,
            "sample_rate": srate,
            "channels_count": n_ch,
            "channels": chs,
            "duration_sec": round(float(n_samp / srate), 2) if srate > 0 else 0.0,
            "n_samples": n_samp
        }

    elif ext == ".csv":
        signals, chs, srate = load_eeg_file(file_path)
        return {
            "file_path": str(p),
            "file_size_bytes": file_size,
            "file_format": ext,
            "sample_rate": srate,
            "channels_count": len(chs),
            "channels": chs,
            "duration_sec": round(float(signals.shape[1] / srate), 2) if srate > 0 else 0.0,
            "n_samples": signals.shape[1]
        }

    raise ValueError(f"Unsupported EEG format for metadata inspection: {ext}")
