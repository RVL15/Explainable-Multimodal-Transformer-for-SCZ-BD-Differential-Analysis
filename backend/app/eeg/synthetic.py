import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from app.config import DEMO_DATA_DIR, STANDARD_16_CHANNELS, DEFAULT_SAMPLE_RATE

def generate_synthetic_channel_signal(
    duration_sec: float,
    sample_rate: float,
    diagnostic_label: str,
    channel_idx: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generates realistic synthetic EEG waveform for a single channel with physiological rhythms.
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    
    # 1/f Pink noise background
    white_noise = rng.standard_normal(n_samples)
    # FFT filtering to make pink noise (1/f)
    fft_vals = np.fft.rfft(white_noise)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    freqs[0] = 1.0  # Avoid divide by zero
    pink_filter = 1.0 / np.sqrt(freqs)
    pink_fft = fft_vals * pink_filter
    pink_noise = np.fft.irfft(pink_fft, n=n_samples)
    pink_noise = (pink_noise / (np.std(pink_noise) + 1e-8)) * 15.0 # ~15 uV scale

    # Canonical oscillations based on diagnosis and channel topography
    # Channel 0, 1 = Fp1, Fp2 (Frontal pole)
    # Channel 8, 9 = O1, O2 (Occipital - dominant alpha)
    is_occipital = channel_idx in [8, 9]
    is_frontal = channel_idx in [0, 1, 2, 3]

    # Delta (1.5 Hz) baseline
    delta = 10.0 * np.sin(2 * np.pi * 1.5 * t + rng.uniform(0, 2*np.pi))

    # Theta rhythm (5.5 Hz)
    theta_amp = 8.0
    if diagnostic_label == "SCZ" and is_frontal:
        theta_amp = 18.0 # Prominent frontal theta in SCZ literature
    elif diagnostic_label == "BD" and is_frontal:
        theta_amp = 11.0
    theta = theta_amp * np.sin(2 * np.pi * 5.5 * t + rng.uniform(0, 2*np.pi))

    # Alpha rhythm (10.0 Hz)
    alpha_amp = 14.0 if is_occipital else 7.0
    if diagnostic_label == "SCZ":
        alpha_amp *= 0.6 # Attenuated alpha in SCZ
    elif diagnostic_label == "BD":
        # Frontal asymmetry
        if channel_idx == 2: # F3 (left)
            alpha_amp *= 0.8
        elif channel_idx == 3: # F4 (right)
            alpha_amp *= 1.3
    alpha = alpha_amp * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2*np.pi))

    # Beta rhythm (22.0 Hz)
    beta_amp = 4.0
    if diagnostic_label == "BD":
        beta_amp = 9.0 # Elevated beta in bipolar spectrum
    beta = beta_amp * np.sin(2 * np.pi * 22.0 * t + rng.uniform(0, 2*np.pi))

    # Gamma rhythm (38.0 Hz)
    gamma_amp = 2.0
    gamma = gamma_amp * np.sin(2 * np.pi * 38.0 * t + rng.uniform(0, 2*np.pi))

    # Occasional eye-blink or muscular transient on frontal channels
    artifact = np.zeros(n_samples)
    if is_frontal and rng.uniform() > 0.4:
        blink_time = rng.integers(int(0.2 * n_samples), int(0.8 * n_samples))
        blink_width = int(0.15 * sample_rate)
        if blink_time + blink_width < n_samples:
            artifact[blink_time : blink_time + blink_width] += rng.uniform(40, 75) * np.hanning(blink_width)

    raw_signal = pink_noise + delta + theta + alpha + beta + gamma + artifact
    return raw_signal.astype(np.float32)

def generate_synthetic_subject_recording(
    subject_id: str,
    diagnostic_label: str,
    duration_sec: float = 30.0,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    seed: int = 42
) -> Tuple[np.ndarray, List[str], float]:
    """
    Generates a full 16-channel synthetic EEG recording clearly labeled [SYNTHETIC DEMO].
    """
    rng = np.random.default_rng(seed)
    channels = STANDARD_16_CHANNELS
    signals = []
    
    for ch_idx, ch_name in enumerate(channels):
        ch_sig = generate_synthetic_channel_signal(
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            diagnostic_label=diagnostic_label,
            channel_idx=ch_idx,
            rng=rng
        )
        signals.append(ch_sig)

    signals_arr = np.array(signals, dtype=np.float32)
    return signals_arr, channels, sample_rate

def populate_demo_cohort(num_scz: int = 20, num_bd: int = 20, num_hc: int = 15) -> List[Dict[str, Any]]:
    """
    Creates demo files on disk and returns subject records for SQLite registration.
    """
    from app.database import get_db
    
    DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    
    specs = [
        ("SCZ", num_scz),
        ("BD", num_bd),
        ("HC", num_hc)
    ]
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        for label, count in specs:
            for i in range(1, count + 1):
                sub_id = f"DEMO_{label}_{i:03d}"
                file_name = f"{sub_id}_eeg.npy"
                file_path = DEMO_DATA_DIR / file_name
                
                # Generate if not exists
                if not file_path.exists():
                    seed = hash(sub_id) % (2**31 - 1)
                    signals, _, _ = generate_synthetic_subject_recording(
                        subject_id=sub_id,
                        diagnostic_label=label,
                        duration_sec=30.0,
                        seed=seed
                    )
                    np.save(str(file_path), signals)
                
                # Check DB
                cursor.execute(
                    "SELECT id FROM subjects WHERE dataset_id = 'demo' AND subject_id = ?",
                    (sub_id,)
                )
                existing = cursor.fetchone()
                
                age = float(np.random.default_rng(i).integers(22, 58))
                sex = "M" if (i % 2 == 0) else "F"
                site = f"Site_{(i % 3) + 1}"
                
                if not existing:
                    cursor.execute("""
                    INSERT INTO subjects (
                        subject_id, dataset_id, diagnostic_label, age, sex, site,
                        recording_path, file_exists, is_excluded, is_synthetic
                    ) VALUES (?, 'demo', ?, ?, ?, ?, ?, 1, 0, 1)
                    """, (sub_id, label, age, sex, site, str(file_path)))
                
                records.append({
                    "subject_id": sub_id,
                    "dataset_id": "demo",
                    "diagnostic_label": label,
                    "age": age,
                    "sex": sex,
                    "site": site,
                    "recording_path": str(file_path),
                    "is_synthetic": True
                })

        # Update demo dataset verified count
        cursor.execute("""
        UPDATE datasets SET
            verified_count = (SELECT COUNT(*) FROM subjects WHERE dataset_id = 'demo' AND is_excluded = 0),
            is_ready = 1,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = 'demo'
        """)

    return records
