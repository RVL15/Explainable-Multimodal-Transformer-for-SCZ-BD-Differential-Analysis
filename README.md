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

The system enforces strict dataset separation and honest clinical task reporting across two distinct classification tasks.

### A. Primary Clinical Objective: SCZ vs. BD Differential Classification
- **Intended Cohort**: B-SNIP 1 (Bipolar & Schizophrenia Network on Intermediate Phenotypes, [NIH NDA Collection #2274](https://nda.nih.gov/edit_collection.html?id=2274))
- **Clinical Task**: Differential classification between Schizophrenia (`SCZ`, class 0) and Bipolar Disorder (`BD`, class 1, probands with psychosis).
- **Legitimate Alternative Data Sources Investigated**:
  1. *NIH NDA Collection #2274 (B-SNIP 1)*: Scalp EEG (64/128-ch) across auditory oddball, P300, and resting state from >1,000 subjects across 5 clinical sites (Baltimore, Boston, Chicago, Dallas, Detroit). Requires NIH eRA Commons account, Data Use Certification (DUC) signed by Principal Investigator and Institutional Signing Official (SO), and documented IRB protocol approval.
  2. *Zenodo Record #14808296*: Resting-state EEG & PANSS tables from schizophrenia patients and healthy controls (ICD-10 F20/F25). Verified count: 50 SCZ, 42 HC. Contains 0 bipolar disorder participants.
  3. *ASEEG Benchmark (Zenodo #18029536)*: Verified count: 47 SCZ, 50 HC. Contains 0 bipolar disorder participants.
  4. *OpenNeuro ds003404*: Multimodal study of visual shape completion in SCZ and BD. Modality is fMRI/MRI, not continuous time-series EEG.
  5. *BD² Integrated Network*: Longitudinal biomarker study with bipolar participants only; lacks schizophrenia comparative arm under identical EEG acquisition protocol.
  6. *BRMH / Park et al. (OSF/Kaggle)*: Tabular quantitative EEG (QEEG) band power and functional connectivity features across 19 channels; lacks continuous raw voltage time-series tensors required by temporal CNN-Transformer architectures.
  7. *Russian Adolescent EEG & ASZED*: Schizophrenia and control populations only; 0 BD participants.
- **Data Status**: Controlled clinical data. Access is legally restricted by federal human-subjects privacy agreements (NIH NDA DUC).
- **Implementation Status**:
  - Ingestion Engine & Data Loader: Fully implemented in `app/datasets/bsnip_manager.py` and `app/datasets/ingestion.py`.
  - Label Mapping & Validation: Explicit binary mapping (`SCZ` = 0, `BD` = 1). Healthy controls (`HC`) are strictly rejected and never substituted for bipolar disorder.
  - Architecture: `EEGCNNTransformer` fully updated to support batched window (3D) and batched subject (4D) forward passes for binary SCZ vs BD classification.
  - Checkpoint Routing: Dedicated checkpoint `backend/checkpoints/scz_bd_model.pt` strictly separated from `backend/checkpoints/best_model.pt`. Real inference enforces checkpoint presence and returns explicit HTTP 400 with authorized access guidance when the checkpoint is absent.
  - Explainability: Integrated Gradients in `app/xai/attribution.py` verified for both target classes (`target_class=0` for SCZ, `target_class=1` for BD).
  - Frontend Integration: Task selector in `AnalyzeEEG.tsx` provides dedicated SCZ vs BD selection, dynamically rendering readiness status and active checkpoint telemetry.

### B. Secondary Benchmark Dataset: ASEEG (SZ vs. HC Technical Validation)
- **Official Source**: Zenodo Record #18029536 ([https://zenodo.org/records/18029536](https://zenodo.org/records/18029536))
- **File Format**: MATLAB v7.3 HDF5 `.mat` files in `backend/data/aseeg/EO/` (`sz/` and `normal/`).
- **Verified Cohort**: Exactly **97 participants** (47 Schizophrenia, 50 Healthy Controls).
- **Montage & Sampling Rate**: 19 electrodes (10–20 system) acquired at 500 Hz, eyes-open (EO) resting state (~76.8 seconds, ~38,400 samples/channel).
- **Designated Task**: Strictly designated for **Schizophrenia vs. Healthy Controls (SZ vs. HC)** technical validation.
- **Active Checkpoint**: `backend/checkpoints/best_model.pt`.
- **Status**: **`MODEL READY`** (Held-out test accuracy: 72.0% unconstrained, 76.2% uncertainty-gated, ROC-AUC 0.8718).

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
4. **Filtering & Referencing**: 4th-order Butterworth bandpass filter (1.0–45.0 Hz) and Common Average Referencing (CAR).
5. **Window Segmentation & Artifact Rejection**: Sliced into 2.0-second non-overlapping windows (500 time points). Flatline or corrupted windows are rejected.
6. **Training-Only Normalization**: Channel-wise mean ($\mu_{\text{train}}$) and standard deviation ($\sigma_{\text{train}}$) are calculated **strictly from training participant windows**. Normalization statistics are saved with the checkpoint and applied to all test and inference recordings.

---

## 4. Trained Model & Genuine Held-Out Evaluation

- **Architecture**: `EEGCNNTransformer` featuring 1D temporal convolutions, spatial feature extraction, 4-head Transformer encoder layers, and attentive subject pooling.
- **Task Mapping**:
  - `SZ_VS_HC`: Checkpoint `backend/checkpoints/best_model.pt`
  - `SCZ_VS_BD`: Checkpoint `backend/checkpoints/scz_bd_model.pt`
- **Checkpoint Contents**: Model state dict, architecture configuration, task mapping, normalization statistics (`norm_mean`, `norm_std`), participant split IDs, and preprocessing metadata.

### Genuine Evaluation on 25 Untouched Test Participants (SZ vs. HC)

Evaluation was performed on the untouched test partition using `python backend/scripts/eval_test_set.py --task SZ_VS_HC`:

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

---

## 5. Live Inference & Explainability Workflow

The website and API provide genuine checkpoint inference:

1. **Supported Formats**: `.mat` (including MATLAB v7.3 HDF5), `.edf`, `.bdf`, `.fif`, `.npy`, and `.csv`.
2. **Channel Validation**: Validates 10–20 electrode presence and duration.
3. **Calibrated Output**: Returns predicted class, probabilities, uncertainty status, and Integrated Gradients electrode attribution rankings.
4. **Task Isolation**: Requests explicitly specify `target_task="SZ_VS_HC"` or `target_task="SCZ_VS_BD"`. If `SCZ_VS_BD` is requested without `scz_bd_model.pt`, the API explicitly rejects the request with HTTP 400 detailing B-SNIP data access requirements and never falls back to the healthy control model.

### Example Inference via PowerShell / cURL
```powershell
# SZ vs HC inference
curl.exe -s -X POST "http://127.0.0.1:8000/api/multimodal/analyze" `
  -F "eeg_file=@backend/data/aseeg/EO/sz/sz2.mat" `
  -F "target_task=SZ_VS_HC" `
  -F "uncertainty_threshold=0.65"

# SCZ vs BD evaluation script
python backend/scripts/eval_test_set.py --task SCZ_VS_BD
```

---

## 6. Verification and Automated Testing

Run the complete 42-test backend verification suite and frontend build:

```powershell
# Run backend test suite (42 unit and integration tests)
python -m pytest backend/tests -v

# Run frontend production build
cd frontend
npm run build
```

---

## 7. Research Limitations & Notice

1. **Not a Clinical Diagnostic Device**: This platform is an engineering research prototype for multimodal explainable AI evaluation. It is not certified by FDA, CE, or CDSCO for clinical diagnosis.
2. **Dataset Scale**: Evaluated on 97 ASEEG participants for SZ vs HC. Multi-site clinical validation is required before diagnostic deployment.
3. **Task Availability**: SCZ vs. BD differential classification requires authorized B-SNIP 1 data and remains disabled until authorized data is provided.

