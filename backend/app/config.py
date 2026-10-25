import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = BASE_DIR / "research.db"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
REPORTS_DIR = BASE_DIR / "reports"
DEMO_DATA_DIR = DATA_DIR / "demo"
MANIFESTS_DIR = DATA_DIR / "manifests"

# Create directories if they do not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

# EEG Signal Processing Constants
DEFAULT_SAMPLE_RATE = 250  # Hz
BANDPASS_LOW_HZ = 1.0     # High-pass filter cutoff
BANDPASS_HIGH_HZ = 45.0   # Low-pass filter cutoff
WINDOW_DURATION_SEC = 2.0 # 2-second windows (500 samples at 250 Hz)
WINDOW_SAMPLES = int(DEFAULT_SAMPLE_RATE * WINDOW_DURATION_SEC)
AMPLITUDE_REJECTION_UV = 150.0 # microvolts threshold for artifact window rejection

# Standard 10-20 EEG 16-channel montage for differential research
STANDARD_16_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"
]

# Primary Dataset Metadata
BSNIP_NDA_COLLECTION_ID = "2274"
BSNIP_NDA_URL = "https://nda.nih.gov/edit_collection.html?id=2274"
BSNIP_TITLE = "Bipolar & Schizophrenia Network on Intermediate Phenotypes (B-SNIP 1)"

# Technical Validation Dataset Metadata
ASEEG_ZENODO_RECORD = "18029536"
ASEEG_ZENODO_URL = "https://zenodo.org/records/18029536"
ASEEG_TITLE = "ASEEG Technical Validation Dataset"

# Model Inference Uncertainty Threshold (0.50 - 1.0)
DEFAULT_UNCERTAINTY_THRESHOLD = 0.65
