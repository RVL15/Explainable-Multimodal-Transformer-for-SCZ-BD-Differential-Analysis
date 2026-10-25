# Explainable Multimodal Transformer for SCZ–BD Differential Analysis

[![Python Version](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-orange.svg)](https://pytorch.org)
[![MNE EEG](https://img.shields.io/badge/MNE-1.13%2B-purple.svg)](https://mne.tools)
[![React](https://img.shields.io/badge/React-19%20%7C%20TypeScript-blue.svg)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5%2B-purple.svg)](https://vitejs.dev)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-v4-38bdf8.svg)](https://tailwindcss.com)

> **Research Platform**: Explainable Multimodal Transformer for SCZ–BD Differential Analysis  
> **Scientific Integrity Standard**: Authentic Dataset Ingestion • Participant-Level Partitioning • Genuine Model Checkpoint Inference • Zero Data Leakage • Prediction-Only UI

---

## 1. Quick Start: Running the Application

The user-facing website is a clean, professional, light-themed **prediction-only application**. All model training, evaluation, and dataset ingestion are handled strictly through background development tools and are not exposed in the user interface.

### A. Start the FastAPI Backend
```powershell
# Navigate to project root
cd "c:\Users\risha\OneDrive\Desktop\FINAL PROJECT"

# Start FastAPI server on port 8000
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```
- Interactive API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### B. Start the React Frontend
```powershell
# Navigate to frontend folder
cd "c:\Users\risha\OneDrive\Desktop\FINAL PROJECT\frontend"

# Start Vite dev server on port 5173
npm run dev -- --host 127.0.0.1 --port 5173
```
- Prediction Application UI: [http://127.0.0.1:5173](http://127.0.0.1:5173)

### C. Run the Complete Automated Verification Suite
```powershell
# Run backend test suite (33 unit and integration tests)
python -m pytest backend/tests -v

# Verify frontend production build
cd frontend
npm run build
```

---

## 2. Dataset Ingestion & Scientific Integrity

The system enforces strict dataset separation and honest clinical task reporting.

### A. Primary Research Objective: SCZ vs. BD (B-SNIP 1)
- **Official Source**: Bipolar & Schizophrenia Network on Intermediate Phenotypes ([NIH NDA Collection #2274](https://nda.nih.gov/edit_collection.html?id=2274))
- **Target Task**: Differential diagnosis between Schizophrenia (SCZ) and Bipolar Disorder with Psychosis (BD).
- **Access Status**: Protected under NIH Data Use Certification (DUC). Requires institutional IRB approval and NIH NDA authorization.
- **Local Availability**: **`NDA_PROTECTED_UNAVAILABLE`** (0 files on disk).
- **Scientific Integrity Guarantee**: The system strictly refuses to fabricate BD labels or substitute healthy controls as bipolar participants. Any inference request targeting `SCZ_VS_BD` returns an HTTP 400 rejection explaining data access requirements. No fake SCZ-vs-BD classifier is claimed.

### B. Secondary Benchmark Dataset: ASEEG (SZ vs. HC)
- **Official Source**: Zenodo Record #18029536 ([https://zenodo.org/records/18029536](https://zenodo.org/records/18029536))
- **File Format**: MATLAB v7.3 HDF5 `.mat` files in `backend/data/aseeg/EO/` (`sz/` and `normal/`).
- **Verified Cohort**: Exactly **97 participants** (47 Schizophrenia, 50 Healthy Controls).
- **Montage & Sampling Rate**: 19 electrodes (10–20 system) acquired at 500 Hz, eyes-open (EO) resting state (~76.8 seconds, ~38,400 samples/channel).
- **Designated Task**: Strictly designated for **Schizophrenia vs. Healthy Controls (SZ vs. HC)** technical validation.

### C. Additional Clinical Dataset: Zenodo Record #14808296
- **Official Source**: Rácz et al., 2025 ([https://zenodo.org/records/14808296](https://zenodo.org/records/14808296))
- **Clinical Tables**: `schizophrenia_EI_dynamics_table.xlsx` (38 SZ, 39 HC) with PANSS scores.
- **Storage Location**: `backend/data/zenodo_14808296/`.

---

## 3. Strict Participant-Level Partitioning & Preprocessing

To eliminate temporal and subject leakage:

1. **Participant-Level Partitioning**: Subjects are partitioned strictly at the subject level before EEG window segmentation. With 97 participants:
   - **Training Set (58.8%)**: 57 participants (28 SZ, 29 HC).
   - **Validation Set (15.5%)**: 15 participants (6 SZ, 9 HC).
   - **Held-Out Test Set (25.8%)**: 25 participants (13 SZ, 12 HC).
   - **Zero Leakage**: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$. Participant IDs assigned to each partition are recorded in `backend/reports/split_manifest.json`.
2. **Channel Alignment**: Channels are mapped to `STANDARD_16_CHANNELS` (`Fp1`, `Fp2`, `F3`, `F4`, `C3`, `C4`, `P3`, `P4`, `O1`, `O2`, `F7`, `F8`, `T3`, `T4`, `T5`, `T6`). Missing channels trigger an informative error rather than silent zero-padding.
3. **Resampling**: Resampled to a standardized `DEFAULT_SAMPLE_RATE = 250` Hz using Fourier anti-aliased interpolation.
4. **Filtering & Referencing**: 4th-order Butterworth bandpass filter (0.5–45.0 Hz) and Common Average Referencing (CAR).
5. **Window Segmentation & Artifact Rejection**: Sliced into 2.0-second non-overlapping windows (500 time points). Flatline or corrupted windows are rejected.
6. **Training-Only Normalization**: Channel-wise mean ($\mu_{\text{train}}$) and standard deviation ($\sigma_{\text{train}}$) are calculated **strictly from training participant windows**. Normalization statistics are saved with the checkpoint and applied to all test and inference recordings.

---

## 4. Trained Model & Genuine Held-Out Evaluation

- **Architecture**: `EEGCNNTransformer` featuring 1D temporal convolutions, spatial feature extraction, 4-head Transformer encoder layers, and attentive subject pooling.
- **Checkpoint Location**: `backend/checkpoints/best_model.pt`.
- **Checkpoint Contents**: Model weights, architecture configuration, task mapping (`SZ_VS_HC`), normalization statistics (`norm_mean`, `norm_std`), participant split IDs, and preprocessing metadata.

### Genuine Evaluation on 25 Untouched Test Participants

Evaluation was performed on the untouched test partition using `backend/scripts/eval_test_set.py`:

#### A. Unconstrained Cohort ($N = 25$)
- **Participants**: 25 (13 SZ, 12 HC)
- **Accuracy**: **72.00%** (18 / 25)
- **Precision**: **68.75%** (11 / 16)
- **Recall / Sensitivity**: **84.62%** (11 / 13)
- **Specificity**: **58.33%** (7 / 12)
- **F1-Score**: **75.86%**
- **ROC-AUC**: **0.8718**
- **Confusion Matrix**:
  $$\begin{bmatrix} 7 & 5 \\ 2 & 11 \end{bmatrix} \quad (\Sigma = 25)$$

#### B. Uncertainty-Gated Cohort (Threshold = 0.65)
- **Coverage**: **84.00%** ($N = 21$ accepted, $N = 4$ uncertain)
- **Uncertain Test Subjects**: `ASEEG_SZ_40`, `ASEEG_HC_24`, `ASEEG_HC_29`, `ASEEG_HC_23`
- **Accuracy on Accepted**: **76.19%** (16 / 21)
- **Precision**: **66.67%** (10 / 15)
- **Recall**: **100.00%** (10 / 10)
- **Specificity**: **54.55%** (6 / 11)
- **F1-Score**: **80.00%**
- **Confusion Matrix**:
  $$\begin{bmatrix} 6 & 5 \\ 0 & 10 \end{bmatrix} \quad (\Sigma = 21)$$

> **Resolution of 20 vs. 25 Discrepancy**: Earlier documentation reported a confusion matrix summing to 20 or 21 because uncertain participants were excluded by the rejection filter without reporting the full cohort denominator ($21 \text{ accepted} + 4 \text{ uncertain} = 25 \text{ total}$). Both the unconstrained and gated metrics are now explicitly documented.

---

## 5. Live Inference & Explainability Workflow

The website and API provide genuine checkpoint inference:

1. **Supported Formats**: `.mat` (including MATLAB v7.3 HDF5), `.edf`, `.bdf`, `.fif`, `.npy`, and `.csv`.
2. **Channel Validation**: Validates 10–20 electrode presence and duration.
3. **Calibrated Output**: Returns predicted class (`SZ` or `HC`), probabilities (`prob_sz`, `prob_hc`), uncertainty status, and Integrated Gradients electrode attribution rankings.

### Example Inference via PowerShell / cURL
```powershell
curl.exe -s -X POST "http://127.0.0.1:8000/api/multimodal/analyze" `
  -F "eeg_file=@backend/data/aseeg/EO/sz/sz2.mat" `
  -F "uncertainty_threshold=0.65"
```

### Example JSON Response
```json
{
  "analysis_id": "pred_345d19c038",
  "subject_id": "UPLOAD_sz2",
  "predicted_label": "SZ",
  "target_task": "SZ_VS_HC",
  "prob_sz": 0.9195,
  "prob_hc": 0.0805,
  "confidence": 0.9195,
  "is_uncertain": false,
  "top_contributing_channels": ["F3", "O2", "P3", "T4"]
}
```

---

## 6. Website Architecture (Prediction-Only)

The user interface is strictly dedicated to prediction workflows:
- **Prediction (`/`)**: EEG upload, format validation, quick sample testing (`n1.mat`, `sz2.mat`), genuine model inference, probability breakdown, and top electrode attributions.
- **Prediction History (`/history`)**: Persisted prediction records retrieved from `backend/research.db` with filtering and CSV export.
- **Explainability (`/explainability`)**: Topographic electrode maps and Integrated Gradients channel saliency rankings.

*Development dashboards, training controls, epoch/optimizer settings, and dataset administration panels are kept strictly in backend development scripts and are not exposed in the web interface.*

---

## 7. Research Limitations & Notice

1. **Not a Clinical Diagnostic Device**: This platform is an engineering research prototype for multimodal explainable AI evaluation. It is not certified by FDA, CE, or CDSCO for clinical diagnosis.
2. **Dataset Scale**: Evaluated on 97 ASEEG participants. External clinical validity requires multi-site validation.
3. **Task Availability**: SCZ vs. BD differential classification requires authorized B-SNIP 1 data and remains disabled until authorized data is provided.
