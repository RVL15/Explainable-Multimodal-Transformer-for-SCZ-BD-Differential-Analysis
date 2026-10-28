# Unseen EEG Data Ingestion & Model Inference Guide

This directory manages **unseen electroencephalography (EEG) recordings** for out-of-sample forward inference using the trained deep learning checkpoints of the **Explainable Multimodal Transformer for Schizophrenia and Bipolar Disorder Differential Analysis** research platform.

---

## 1. Directory Structure

```text
backend/data/unseen/
├── README.md              # Detailed documentation and execution protocol
├── incoming/              # Place newly acquired, out-of-sample raw/preprocessed EEG recordings here
│   └── .gitkeep
├── processed/             # Optional storage for intermediate window tensors (num_windows, 16, 500)
│   └── .gitkeep
└── results/               # Structured JSON prediction outputs, uncertainty telemetry, and XAI rankings
    └── .gitkeep
```

---

## 2. What "Unseen EEG Data" Means

In neurocomputational machine learning, **unseen EEG data** refers to clinical or synthetic electrophysiological recordings collected from participants who **were never included** in the model's training, validation, or hyperparameter selection splits:
- **No Temporal Leakage**: Windows are segmented only after complete participant-level isolation.
- **No Subject Leakage**: The participant identifier is not part of the training cohort.
- **No Feature / Normalization Leakage**: Normalization mean ($\mu$) and standard deviation ($\sigma$) are **not** recalculated from the incoming recording. Instead, the model strictly applies the frozen statistics saved within the trained checkpoint (`norm_mean`, `norm_std`).
- **Zero Retraining**: Forward inference executes in frozen evaluation mode (`model.eval()`, `torch.no_grad()`). Model weights and checkpoints are never modified during or after inference.

---

## 3. Supported Input Formats & Requirements

Recordings placed into `backend/data/unseen/incoming/` are validated by the neurophysiological inspection engine before inference:

| Format | Extension | Reader / Parser Engine | Minimum Requirements |
| :--- | :---: | :--- | :--- |
| **MATLAB File** | `.mat` | SciPy `loadmat` (v5/v7) or `h5py` (v7.3 HDF5) | 2D matrix (channels $\times$ time or transposed) |
| **European Data Format** | `.edf` / `.bdf` | MNE-Python `read_raw_edf` / `read_raw_bdf` | Continuous voltage time-series |
| **Neuromag / Elekta FIF**| `.fif` | MNE-Python `read_raw_fif` | Continuous raw EEG |
| **NumPy Array** | `.npy` | NumPy `np.load` | 2D array of shape `(16, n_times)` or `(n_times, 16)` |
| **Tabular Matrix** | `.csv` | Pandas `pd.read_csv` | Standard channels as headers or numeric rows |

### Neurophysiological Requirements:
1. **Electrode Montage**: Must contain standard 10–20 electrode channels. The core 16-channel montage required by the model encoder is:
   `Fp1`, `Fp2`, `F3`, `F4`, `C3`, `C4`, `P3`, `P4`, `O1`, `O2`, `F7`, `F8`, `T3`, `T4`, `T5`, `T6`.
   - Missing required channels trigger an explicit validation error. Missing channels are **never** zero-filled or synthesized.
2. **Sampling Rate**: Minimum 50 Hz. Signals are automatically anti-aliased and resampled to the standardized `DEFAULT_SAMPLE_RATE = 250 Hz`.
3. **Recording Duration**: Minimum 2.0 seconds (500 time points at 250 Hz) to allow at least one valid analysis window.
4. **Dynamic Range & Artifacts**: Clean dynamic range with amplitude rejection threshold at $\pm 150\,\mu\text{V}$ and peak-to-peak threshold at $300\,\mu\text{V}$. Flatline recordings ($< 0.1\,\mu\text{V}$) are automatically rejected.

---

## 4. Checkpoints & Task Separation

The research platform enforces strict separation between two distinct clinical classification tasks:

### Task A: Schizophrenia vs. Healthy Controls (`SZ_VS_HC`)
- **Clinical Task**: Differentiate Schizophrenia (`SZ`, class 0) from Healthy Controls (`HC`, class 1).
- **Dataset**: ASEEG Technical Validation Benchmark (Zenodo #18029536, 97 participants: 47 SZ, 50 HC).
- **Active Checkpoint**: `backend/checkpoints/best_model.pt`.
- **Status**: **`MODEL READY`** (Held-out test accuracy: 72.0% unconstrained, 76.2% uncertainty-gated, ROC-AUC 0.8718).
- **Output Classes**: `SZ` (Schizophrenia) and `HC` (Healthy Control).

### Task B: Schizophrenia vs. Bipolar Disorder (`SCZ_VS_BD`)
- **Clinical Task**: Differential diagnosis between Schizophrenia (`SCZ`, class 0) and Bipolar Disorder (`BD`, class 1, probands with psychosis).
- **Intended Dataset**: NIH NDA Collection #2274 (B-SNIP 1).
- **Dedicated Checkpoint**: `backend/checkpoints/scz_bd_model.pt`.
- **Data Status**: Controlled clinical human-subjects data requiring NIH eRA Commons credentials and signed Data Use Certification (DUC).
- **Integrity Rule**: `best_model.pt` is **never** substituted for this task. If `scz_bd_model.pt` is missing, inference returns a clear, honest error explaining that authorized B-SNIP 1 clinical data is required. Healthy controls (`HC`) are never mislabeled as bipolar disorder (`BD`).

---

## 5. How to Run Inference

### Method A: Via Command Line (Batch or Single File)
Ensure you are in the project root:
```powershell
cd "c:\Users\risha\OneDrive\Desktop\FINAL PROJECT"
```

1. **Process a specific EEG file**:
   ```powershell
   python backend/scripts/run_unseen_inference.py --file backend/data/unseen/incoming/my_recording.mat --task SZ_VS_HC --threshold 0.65
   ```

2. **Process all files currently inside `backend/data/unseen/incoming/`**:
   ```powershell
   python backend/scripts/run_unseen_inference.py --task SZ_VS_HC
   ```

3. **Optional: Save preprocessed arrays in `processed/`**:
   ```powershell
   python backend/scripts/run_unseen_inference.py --file my_recording.edf --save-processed
   ```

### Method B: Via the FastAPI Backend API
1. Start the server (or use `run.bat`):
   ```powershell
   python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
   ```
2. **List incoming unseen recordings**:
   ```http
   GET http://127.0.0.1:8000/api/signals/unseen/incoming
   ```
3. **Execute unseen inference**:
   ```http
   POST http://127.0.0.1:8000/api/signals/unseen/predict?filename=my_recording.mat&target_task=SZ_VS_HC&uncertainty_threshold=0.65
   ```
4. **List saved prediction results**:
   ```http
   GET http://127.0.0.1:8000/api/signals/unseen/results
   ```

### Method C: Through the Prediction Application UI
1. Launch the application with `run.bat` or navigate to `http://127.0.0.1:5173`.
2. Navigate to the **Prediction Application** tab.
3. Drag and drop any `.mat`, `.edf`, `.fif`, `.npy`, or `.csv` EEG recording into the upload zone.
4. Select the target classification task and adjust the uncertainty threshold (default: 0.65).
5. Click **Run Multimodal Differential Analysis**.
6. Prediction results, confidence bars, and electrode importance attributions will display immediately and be saved automatically to `backend/data/unseen/results/`.

---

## 6. Prediction Output Format

Every inference run generates a structured JSON report saved in `backend/data/unseen/results/` with the filename format:
`YYYYMMDD_HHMMSS_<filename_stem>_result.json`:

```json
{
  "filename": "subject_sample_01.mat",
  "task": "SZ_VS_HC",
  "predicted_class": "SZ",
  "class_scores": {
    "SZ": 0.7824,
    "HC": 0.2176
  },
  "uncertainty_status": "CONFIDENT",
  "model_checkpoint": "best_model.pt",
  "is_new_input": true,
  "retrained_during_inference": false,
  "explanation_available": true,
  "predicted_label": "SZ",
  "raw_label": "SZ",
  "confidence": 0.7824,
  "uncertainty_threshold": 0.65,
  "is_uncertain": false,
  "usable_windows": 38,
  "total_duration_sec": 76.0,
  "sampling_rate": 500.0,
  "channels_analyzed": 16,
  "top_contributing_channels": ["Fp1", "F3", "C3", "P3"],
  "eeg_channel_attributions": [
    { "channel": "Fp1", "importance_score": 0.1245, "rank": 1 },
    { "channel": "F3",  "importance_score": 0.0982, "rank": 2 },
    { "channel": "C3",  "importance_score": 0.0841, "rank": 3 },
    { "channel": "P3",  "importance_score": 0.0763, "rank": 4 }
  ],
  "scientific_disclaimer": "RESEARCH LIMITATION NOTICE: Gradient-based feature attributions reflect mathematical sensitivity of the neural network's trained decision surface. Attribution saliency does NOT imply neurobiological causation, verified physiological biomarkers, or clinical etiology. This prototype is for engineering and scientific evaluation only.",
  "timestamp": "20261028_101500"
}
```

---

## 7. Scientific Integrity & Data Ethics Notice

1. **Zero Retraining**: Incoming recordings are processed strictly in forward-pass evaluation mode. Under no circumstances does the platform modify trained weights, update running gradients, or fine-tune checkpoints during unseen inference.
2. **No Data Leakage**: Normalization parameters (`norm_mean`, `norm_std`) are fixed from the training split. Never calculate standard scaler metrics from unseen or test datasets.
3. **No Task Conflation**: An SZ-vs-HC prediction determines whether an EEG matches schizophrenia patterns versus healthy non-clinical patterns. It provides zero evidence regarding bipolar disorder.
4. **Data Privacy & Consent**: All EEG recordings placed into `incoming/` must be fully de-identified and stripped of direct protected health information (PHI) in compliance with HIPAA Safe Harbor and GDPR standards. Git policies strictly ignore all raw files in `incoming/`, `processed/`, and `results/`.
