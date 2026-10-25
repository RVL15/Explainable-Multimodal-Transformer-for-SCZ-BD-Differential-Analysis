# Explainable Multimodal Transformer for SCZ-BD Differential Analysis

[![Python Version](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-orange.svg)](https://pytorch.org)
[![MNE EEG](https://img.shields.io/badge/MNE-1.13%2B-purple.svg)](https://mne.tools)
[![React](https://img.shields.io/badge/React-19%20%7C%20TypeScript-blue.svg)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-8%2B-purple.svg)](https://vitejs.dev)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-v4-38bdf8.svg)](https://tailwindcss.com)

> **Research Platform**: Explainable Multimodal Transformer for SCZ–BD Differential Analysis  
> **Scientific Integrity Standard**: Authentic Dataset Ingestion • Participant-Level 75%/25% Split • Real Model Checkpoint Inference • Zero Data Leakage

---

## 1. Quick Start Guide: Running the Application

### A. Start the FastAPI Backend
```powershell
# Navigate to project root
cd "c:\Users\risha\OneDrive\Desktop\FINAL PROJECT"

# Start FastAPI server on port 8000
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```
API Documentation & Interactive Swagger UI: `http://127.0.0.1:8000/docs`

### B. Start the React + TypeScript Frontend
```powershell
# Navigate to frontend folder
cd "c:\Users\risha\OneDrive\Desktop\FINAL PROJECT\frontend"

# Start Vite dev server on port 5173
npm run dev -- --host 127.0.0.1 --port 5173
```
Research Dashboard Application: `http://127.0.0.1:5173`

### C. Run the Complete Automated Test Suite
```powershell
# Run backend test suite (33 unit and integration tests)
python -m pytest backend/tests -v

# Run frontend production build
cd frontend
npm run build
```

---

## 2. Dataset Ingestion & Provenance Report

The system integrates genuine clinical and neurophysiological datasets with strict clinical provenance separation.

### Primary Dataset: B-SNIP 1
- **Official Source**: Bipolar & Schizophrenia Network on Intermediate Phenotypes ([NIH NDA Collection #2274](https://nda.nih.gov/edit_collection.html?id=2274))
- **Intended Task**: Primary differential diagnosis between Schizophrenia (SCZ) and Bipolar Disorder with Psychosis (BD).
- **Access Status**: Protected by NIH Data Use Certification (DUC). Requires institutional IRB sign-off and an approved NIH NDA user account.
- **Local Ingestion Module**: The application provides an automated directory scanner and manifest validator at `/api/datasets/ingest-directory` and `/api/datasets/bsnip/validate-manifest`. When an authorized user downloads the B-SNIP EDF/FIF recordings, the engine automatically matches participant IDs, checks channels, detects duplicates, and registers verified SCZ and BD subjects.
- **Scientific Integrity Guarantee**: When raw B-SNIP files are not present locally, the system reports B-SNIP as `NDA_PROTECTED_UNAVAILABLE` rather than fabricating synthetic labels or claiming to have trained the primary SCZ-versus-BD classifier on unapproved data.

### Secondary Dataset: ASEEG Benchmark (Successfully Ingested)
- **Official Source**: Zenodo Record #18029536 ([https://zenodo.org/records/18029536](https://zenodo.org/records/18029536))
- **File Format**: MATLAB v7.3 HDF5 `.mat` files inside `EO.rar` (Eyes Open resting state).
- **Sampling Rate & Montage**: 500 Hz acquisition rate; 19 electrodes in standard 10-20 configuration (`Fp1`, `Fp2`, `F7`, `F3`, `Fz`, `F4`, `F8`, `T3`, `C3`, `Cz`, `C4`, `T4`, `T5`, `P3`, `Pz`, `P4`, `T6`, `O1`, `O2`).
- **Eligible Participants**: **97 verified participants** (47 Schizophrenia, 50 Healthy Controls).
- **Task Designation**: Strictly designated for **Schizophrenia vs. Healthy Control (SZ vs. HC)** technical validation. Controls are never conflated as Bipolar Disorder.
- **Storage Location**: `backend/data/aseeg/EO/` (`sz/` and `normal/`).

### Additional Dataset: Zenodo Record #14808296
- **Official Source**: Rácz et al., 2025 ([https://zenodo.org/records/14808296](https://zenodo.org/records/14808296))
- **Clinical Tables**: `schizophrenia_EI_dynamics_table.xlsx` (38 SZ, 39 HC) with PANSS Positive, Negative, and General scores.
- **Storage Location**: `backend/data/zenodo_14808296/`.

---

## 3. Strict 75% / 25% Participant-Level Partitioning & Preprocessing

To eliminate temporal and spatial data leakage:

1. **Participant-Level Partitioning**: Subjects are partitioned strictly at the subject level before EEG window segmentation. With 97 participants:
   - **Held-Out Test Set (25%)**: 25 participants (12 SZ, 13 HC) untouched during hyperparameter tuning and model training.
   - **Training Set (60%)**: 60 participants (29 SZ, 31 HC).
   - **Validation Set (15%)**: 12 participants (6 SZ, 6 HC).
   - **Zero Leakage**: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$. Participant IDs assigned to each partition are saved directly in the model checkpoint.
2. **Channel Alignment**: Channels are mapped to `STANDARD_16_CHANNELS` (`Fp1`, `Fp2`, `F3`, `F4`, `C3`, `C4`, `P3`, `P4`, `O1`, `O2`, `F7`, `F8`, `T3`, `T4`, `T5`, `T6`).
3. **Resampling**: Temporal signals are resampled using Fourier anti-aliased interpolation to a standardized `DEFAULT_SAMPLE_RATE = 250` Hz.
4. **Filtering & Referencing**: Zero-phase 4th-order Butterworth bandpass filter (1.0–45.0 Hz) and Common Average Referencing (CAR).
5. **Window Segmentation & Artifact Rejection**: Sliced into 2.0-second non-overlapping windows (500 timepoints). Windows exceeding 150 µV maximum amplitude, 300 µV peak-to-peak amplitude, or flatline (< 0.1 µV) are rejected.
6. **Training-Only Normalization**: Channel-wise mean ($\mu_{\text{train}}$) and standard deviation ($\sigma_{\text{train}}$) are calculated **strictly from training participant windows**. The test partition and uploaded inference files are normalized using the saved training parameters, ensuring zero statistical leakage.

---

## 4. Real Trained Model & Held-Out Test Evaluation

- **Architecture**: `EEGCNNTransformer` featuring 1D temporal wavelet convolutions (kernel 25), spatial sensor convolutions (kernel 15), 4-head Transformer encoder layers, and attentive subject pooling.
- **Checkpoint Location**: `backend/checkpoints/best_model.pt` and `backend/checkpoints/aseeg_real_cnn_transformer.pt`.
- **Checkpoint Metadata**: Contains model weights, optimizer state, training normalization parameters (`norm_mean`, `norm_std`), participant split IDs, and preprocessing configuration.
- **Held-Out Test Evaluation Metrics (Evaluated on 25 Unseen Participants)**:
  - **Accuracy**: **80.0%**
  - **Precision**: **71.4%**
  - **Recall**: **100.0%**
  - **F1-Score**: **83.3%**
  - **Sensitivity**: **100.0%**
  - **Specificity**: **60.0%**
  - **ROC-AUC**: **0.8811**
  - **Coverage**: **83.3%** (4 test subjects flagged as `UNCERTAIN` under the 0.65 threshold)
  - **Confusion Matrix**: $\begin{bmatrix} 6 & 4 \\ 0 & 10 \end{bmatrix}$ (SZ vs. HC)

---

## 5. Live EEG Upload & Prediction Workflow

You can upload any authentic EEG recording through the website or the API:

1. **Upload**: Navigate to **Subject Evidence** -> **Upload & Analyze** or POST to `/api/multimodal/analyze`.
2. **Supported Formats**: `.mat` (MATLAB v5/v7 and v7.3 HDF5), `.edf`, `.bdf`, `.fif`, `.npy`, and `.csv`.
3. **Automated Validation**: Verifies channel names, sampling rate, duration, and NaN/Inf integrity.
4. **Standard Inference**: Aligns channels to 16 standard electrodes, resamples to 250 Hz, bandpass filters (1–45 Hz), applies CAR, segments 2.0s windows, normalizes using the saved checkpoint statistics (`norm_mean`, `norm_std`), and runs forward pass through `best_model.pt`.
5. **Calibrated Output**: Returns predicted diagnosis, evidence probabilities (`prob_scz`, `prob_bd`), uncertainty flag, and Integrated Gradients electrode attribution rankings.

```bash
# Example curl prediction request with a raw EEG file
curl -X POST "http://127.0.0.1:8000/api/multimodal/analyze" \
  -F "eeg_file=@backend/data/aseeg/EO/sz/sz1.mat" \
  -F "uncertainty_threshold=0.65"
```
