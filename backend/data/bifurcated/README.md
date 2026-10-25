# Bifurcated EEG Research Data Repository

This directory organizes and bifurcates the EEG dataset into clear, categorized folders for testing and validation.

## Folder Structure

```
backend/data/bifurcated/
├── validated/
│   ├── scz/                     # 47 validated Schizophrenia EEG recordings (sz1.mat - sz51.mat)
│   └── not_scz/                 # 50 validated Healthy Control / Normal EEG recordings (n1.mat - n50.mat)
├── testing/                     # Held-out 25-participant test set (zero data leakage)
│   ├── scz/                     # 12 unseen Schizophrenia test recordings
│   └── not_scz/                 # 13 unseen Healthy Control test recordings
├── not_validated/               # Files excluded from direct neural network tensor input
│   ├── raw_archives/            # Compressed archive files (e.g. EO.rar, 247 MB)
│   └── tabular_clinical_tables/ # Clinical demographic tables lacking raw EEG time series
└── bifurcation_manifest.csv     # Master inventory of all recordings with validation status
```

## Folder Descriptions

1. **`validated/scz/` (47 files)**
   - Genuine Schizophrenia participant EEG recordings.
   - 19 scalp channels, 500 Hz sampling rate, 76.8s continuous duration.
   - Pass automated signal validation (non-empty, finite values, standard 10-20 compliance).

2. **`validated/not_scz/` (50 files)**
   - Genuine Healthy Control participant EEG recordings.
   - Strictly labeled as Healthy Controls (HC). Never conflated with Bipolar Disorder (BD).
   - 19 scalp channels, 500 Hz sampling rate, 76.8s continuous duration.

3. **`testing/scz/` (12 files) & `testing/not_scz/` (13 files)**
   - The 25 held-out test participants partition from `best_model.pt`.
   - Never seen during training or validation loss optimization.
   - Used for evaluating empirical model performance (Accuracy 72.0%, F1 0.6667, ROC-AUC 0.8718).

4. **`not_validated/`**
   - Contains raw archives (such as `.rar` compressed files) and tabular metadata sheets that cannot be directly ingested as multi-channel time-series tensors without prior extraction or paired raw signal files.
