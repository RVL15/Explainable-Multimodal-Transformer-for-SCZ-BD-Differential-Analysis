import numpy as np
from scipy import signal
from typing import Dict, List, Any, Tuple

FREQUENCY_BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

def compute_psd(
    data: np.ndarray,
    sample_rate: float,
    nperseg: int = 250
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Welch's Power Spectral Density.
    data: (num_channels, num_samples) or (num_samples,)
    Returns:
        freqs: 1D array of frequencies up to 50 Hz
        psd: array of power spectral densities
    """
    freqs, psd = signal.welch(
        data,
        fs=sample_rate,
        nperseg=min(nperseg, data.shape[-1]),
        axis=-1
    )
    # Clip to 0 - 50 Hz for focused EEG analysis
    mask = (freqs >= 0.5) & (freqs <= 50.0)
    return freqs[mask], psd[..., mask]

def compute_band_powers(
    freqs: np.ndarray,
    psd: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Calculates absolute and relative power for each canonical EEG frequency band.
    psd: 1D array of power spectral density for a single channel
    """
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    total_power = float(np.sum(psd) * df)
    if total_power <= 0:
        total_power = 1e-9

    band_powers = {}
    for band_name, (low, high) in FREQUENCY_BANDS.items():
        band_mask = (freqs >= low) & (freqs < high)
        if np.any(band_mask):
            abs_power = float(np.sum(psd[band_mask]) * df)
        else:
            abs_power = 0.0
        
        rel_power = float(abs_power / total_power)
        band_powers[band_name] = {
            "freq_range": f"{low}-{high} Hz",
            "absolute_power": round(abs_power, 4),
            "relative_power": round(rel_power, 4)
        }

    return band_powers

def extract_subject_spectral_features(
    data: np.ndarray,
    sample_rate: float
) -> np.ndarray:
    """
    Extracts handcrafted spectral feature vector for the baseline Logistic Regression model.
    Includes relative powers for all 5 bands across channels + Theta/Alpha ratio + Frontal asymmetry.
    Input: data of shape (num_channels, num_samples)
    Output: 1D feature vector
    """
    num_channels = data.shape[0]
    freqs, psd = compute_psd(data, sample_rate)
    
    features = []
    
    # Per-channel relative band powers
    for ch in range(num_channels):
        ch_psd = psd[ch]
        bp = compute_band_powers(freqs, ch_psd)
        for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
            features.append(bp[band]["relative_power"])
        
        # Theta / Alpha ratio (hallmark neuro-indicator)
        alpha_p = max(bp["Alpha"]["relative_power"], 1e-5)
        theta_p = bp["Theta"]["relative_power"]
        features.append(theta_p / alpha_p)
        
    # Frontal asymmetry if at least 4 channels present (e.g. Fp1 vs Fp2, F3 vs F4)
    if num_channels >= 4:
        # ln(Alpha right) - ln(Alpha left)
        bp_left = compute_band_powers(freqs, psd[0])
        bp_right = compute_band_powers(freqs, psd[1])
        left_alpha = max(bp_left["Alpha"]["absolute_power"], 1e-6)
        right_alpha = max(bp_right["Alpha"]["absolute_power"], 1e-6)
        asymmetry = float(np.log(right_alpha) - np.log(left_alpha))
        features.append(asymmetry)
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float32)
