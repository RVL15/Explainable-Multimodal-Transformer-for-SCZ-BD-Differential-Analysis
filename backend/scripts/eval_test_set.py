import sys
sys.path.append("backend")
import os
import json
import torch
import numpy as np
from app.eeg.io import load_eeg_file
from app.models.pipeline import preprocess_and_window_eeg
from app.models.cnn_transformer import EEGCNNTransformer
from app.evaluation.metrics import compute_classification_metrics
from app.database import get_db

ckpt = torch.load("backend/checkpoints/best_model.pt", map_location="cpu")
model = EEGCNNTransformer(num_channels=16, time_points=500, num_classes=2, d_model=64)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

norm_mean = np.array(ckpt["norm_mean"], dtype=np.float32)
norm_std = np.array(ckpt["norm_std"], dtype=np.float32)

test_sub_ids = ckpt["split_participant_ids"]["test"]
print(f"Number of test participants: {len(test_sub_ids)}")

with get_db() as conn:
    c = conn.cursor()
    placeholders = ",".join(["?"] * len(test_sub_ids))
    c.execute(f"SELECT subject_id, diagnostic_label, recording_path FROM subjects WHERE subject_id IN ({placeholders})", test_sub_ids)
    sub_map = {r[0]: dict(r) for r in c.fetchall()}

# Test evaluation with NO uncertainty rejection (threshold = 0.5)
y_true_all = []
y_pred_all = []
y_scores_all = []
results_all = []

# Test evaluation with standard uncertainty rejection (threshold = 0.65)
y_pred_thresh = []
uncertain_subs = []

for sid in test_sub_ids:
    sub = sub_map[sid]
    sig, chs, fs = load_eeg_file(sub["recording_path"])
    windows = preprocess_and_window_eeg(sig, chs, fs)
    norm_w = (windows - norm_mean) / norm_std
    tensor_w = torch.from_numpy(norm_w).float()

    res_raw = model.predict_with_uncertainty(tensor_w, uncertainty_threshold=0.50, target_task="SZ_VS_HC")
    res_thresh = model.predict_with_uncertainty(tensor_w, uncertainty_threshold=0.65, target_task="SZ_VS_HC")

    true_idx = 0 if sub["diagnostic_label"] == "SZ" else 1
    raw_pred_idx = 0 if res_raw["raw_label"] == "SZ" else 1
    thresh_pred_idx = -1 if res_thresh["is_uncertain"] else raw_pred_idx

    y_true_all.append(true_idx)
    y_pred_all.append(raw_pred_idx)
    y_scores_all.append(res_raw["prob_hc"]) # class 1 score

    y_pred_thresh.append(thresh_pred_idx)
    if res_thresh["is_uncertain"]:
        uncertain_subs.append(sid)

    results_all.append({
        "subject_id": sid,
        "true_label": sub["diagnostic_label"],
        "predicted_label": res_thresh["predicted_label"],
        "raw_label": res_raw["raw_label"],
        "prob_sz": res_raw["prob_sz"],
        "prob_hc": res_raw["prob_hc"],
        "confidence": res_raw["confidence"],
        "is_uncertain": res_thresh["is_uncertain"]
    })

metrics_all = compute_classification_metrics(y_true_all, y_pred_all, y_scores_all)
metrics_thresh = compute_classification_metrics(y_true_all, y_pred_thresh, y_scores_all)

print("\n=== METRICS WITHOUT REJECTION (threshold=0.50, all 25 test subjects) ===")
print("Accuracy:", metrics_all["accuracy"])
print("Precision:", metrics_all["precision"])
print("Recall / Sensitivity:", metrics_all["recall"])
print("Specificity:", metrics_all["specificity"])
print("F1 Score:", metrics_all["f1_score"])
print("ROC-AUC:", metrics_all["roc_auc"])
print("Confusion Matrix [[TN, FP], [FN, TP]]:", metrics_all["confusion_matrix"])
print("Confusion Matrix sum:", sum(sum(r) for r in metrics_all["confusion_matrix"]))

print("\n=== METRICS WITH UNCERTAINTY REJECTION (threshold=0.65) ===")
print("Accuracy on accepted:", metrics_thresh["accuracy"])
print("Precision on accepted:", metrics_thresh["precision"])
print("Recall on accepted:", metrics_thresh["recall"])
print("Specificity on accepted:", metrics_thresh["specificity"])
print("F1 Score on accepted:", metrics_thresh["f1_score"])
print("ROC-AUC:", metrics_thresh["roc_auc"])
print("Uncertain count:", metrics_thresh["uncertain_count"])
print("Coverage:", metrics_thresh["coverage"])
print("Confusion Matrix on accepted [[TN, FP], [FN, TP]]:", metrics_thresh["confusion_matrix"])
print("Accepted count in matrix:", sum(sum(r) for r in metrics_thresh["confusion_matrix"]))
print("Uncertain participants:", uncertain_subs)
