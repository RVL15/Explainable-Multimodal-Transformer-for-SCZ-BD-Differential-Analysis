import numpy as np
from scipy import signal
from typing import Tuple, List, Dict, Any, Optional

def apply_bandpass_filter(
    data: np.ndarray,
    sample_rate: float,
    low_freq: float = 1.0,
    high_freq: float = 45.0,
    order: int = 4
) -> np.ndarray:
    """
    Applies zero-phase Butterworth bandpass filter across all channels.
    data: (num_channels, num_samples)
    """
    nyquist = 0.5 * sample_rate
    low = max(0.1, low_freq) / nyquist
    high = min(nyquist - 0.5, high_freq) / nyquist
    
    if low >= high:
        return data

    b, a = signal.butter(order, [low, high], btype='bandpass')
    filtered = signal.filtfilt(b, a, data, axis=-1)
    return filtered

def apply_average_rereference(data: np.ndarray) -> np.ndarray:
    """
    Applies Common Average Reference (CAR).
    data: (num_channels, num_samples)
    """
    avg = np.mean(data, axis=0, keepdims=True)
    return data - avg

def normalize_channels(data: np.ndarray) -> np.ndarray:
    """
    Applies channel-wise z-score normalization.
    data: (num_channels, num_samples)
    """
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std[std == 0] = 1.0
    return (data - mean) / std

def segment_and_reject_artifacts(
    data: np.ndarray,
    sample_rate: float,
    window_sec: float = 2.0,
    overlap_ratio: float = 0.0,
    rejection_threshold_uv: float = 150.0
) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
    """
    Slices multi-channel EEG into fixed time windows and identifies artifact segments.
    Returns:
        clean_windows: np.ndarray of shape (num_clean_windows, num_channels, window_samples)
        rejected_indices: list of rejected window indices
        stats: dict containing total_windows, rejected_windows, rejection_ratio
    """
    num_channels, total_samples = data.shape
    window_samples = int(sample_rate * window_sec)
    step = int(window_samples * (1.0 - overlap_ratio))
    if step < 1:
        step = window_samples

    windows = []
    rejected_indices = []
    
    current_idx = 0
    window_id = 0
    
    while current_idx + window_samples <= total_samples:
        segment = data[:, current_idx : current_idx + window_samples]
        
        # Artifact detection criteria: peak-to-peak amplitude or excessive voltage
        peak_to_peak = np.ptp(segment, axis=-1)
        max_abs = np.max(np.abs(segment), axis=-1)
        
        # If any channel exceeds threshold or is flatline
        is_artifact = np.any(peak_to_peak > (rejection_threshold_uv * 2)) or np.any(max_abs > rejection_threshold_uv) or np.any(peak_to_peak < 0.1)
        
        if is_artifact:
            rejected_indices.append(window_id)
        else:
            windows.append(segment)
            
        current_idx += step
        window_id += 1

    total_windows = window_id
    rejected_count = len(rejected_indices)
    rejection_ratio = float(rejected_count / total_windows) if total_windows > 0 else 0.0

    if len(windows) > 0:
        clean_windows_arr = np.array(windows, dtype=np.float32)
    else:
        # Fallback: if all windows were rejected due to strictness, take original normalized windows
        clean_windows_arr = np.empty((0, num_channels, window_samples), dtype=np.float32)

    stats = {
        "total_windows": total_windows,
        "rejected_windows": rejected_count,
        "clean_windows": len(windows),
        "rejection_ratio": round(rejection_ratio, 4)
    }

    return clean_windows_arr, rejected_indices, stats

def run_eeg_preprocessing(
    raw_data: np.ndarray,
    sample_rate: float,
    bandpass: bool = True,
    low_hz: float = 1.0,
    high_hz: float = 45.0,
    car: bool = True,
    normalize: bool = True,
    window_sec: float = 2.0,
    rejection_threshold_uv: float = 150.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Executes the standard full preprocessing pipeline on raw EEG data.
    Returns:
        processed_full_signal: (num_channels, num_samples)
        clean_windows: (num_clean_windows, num_channels, window_samples)
        rejection_stats: Dict with rejection counts and ratios
    """
    signal = raw_data.copy()

    # 1. Bandpass filter
    if bandpass:
        signal = apply_bandpass_filter(signal, sample_rate, low_hz, high_hz)

    # 2. Average rereferencing
    if car:
        signal = apply_average_rereference(signal)

    # 3. Artifact segmentation before final z-score (using microvolts for rejection)
    clean_windows, rejected_indices, stats = segment_and_reject_artifacts(
        signal,
        sample_rate,
        window_sec=window_sec,
        rejection_threshold_uv=rejection_threshold_uv
    )

    # 4. Normalization
    if normalize:
        signal = normalize_channels(signal)
        if len(clean_windows) > 0:
            # Normalize each clean window
            clean_windows = (clean_windows - np.mean(clean_windows, axis=-1, keepdims=True)) / np.maximum(
                np.std(clean_windows, axis=-1, keepdims=True), 1e-6
            )

    return signal, clean_windows, stats
