import requests
import json

print("=== 1. SYSTEM HEALTH ===")
h = requests.get("http://127.0.0.1:8000/api/health").json()
print("Backend Status:", h["status"])
sz_info = h["classification_tasks_readiness"]["sz_vs_hc"]
bd_info = h["classification_tasks_readiness"]["scz_vs_bd"]
print(f"Task SZ vs HC: {sz_info['status']} | Accuracy: {sz_info['held_out_accuracy']} | Checkpoint: {sz_info['active_checkpoint']}")
print(f"Task SCZ vs BD: {bd_info['status']} | Note: {bd_info['status_reason']}")

print("\n=== 2. INFERENCE: SCHIZOPHRENIA RECORDING (sz2.mat) ===")
with open("backend/data/aseeg/EO/sz/sz2.mat", "rb") as f:
    res = requests.post("http://127.0.0.1:8000/api/multimodal/analyze", files={"eeg_file": ("sz2.mat", f)}, data={"uncertainty_threshold": 0.65})
    sz_out = res.json()
    print(f"Prediction: {sz_out['predicted_label']} | Confidence: {sz_out['confidence']*100:.1f}%")
    print(f"Prob SZ: {sz_out['prob_sz']*100:.1f}% | Prob HC: {sz_out['prob_hc']*100:.1f}% | Uncertain: {sz_out['is_uncertain']}")
    print(f"Top 4 Attribution Electrodes: {sz_out['top_contributing_channels'][:4]}")

print("\n=== 3. INFERENCE: HEALTHY CONTROL RECORDING (n1.mat) ===")
with open("backend/data/aseeg/EO/normal/n1.mat", "rb") as f:
    res = requests.post("http://127.0.0.1:8000/api/multimodal/analyze", files={"eeg_file": ("n1.mat", f)}, data={"uncertainty_threshold": 0.65})
    hc_out = res.json()
    print(f"Prediction: {hc_out['predicted_label']} | Confidence: {hc_out['confidence']*100:.1f}%")
    print(f"Prob SZ: {hc_out['prob_sz']*100:.1f}% | Prob HC: {hc_out['prob_hc']*100:.1f}% | Uncertain: {hc_out['is_uncertain']}")
    print(f"Top 4 Attribution Electrodes: {hc_out['top_contributing_channels'][:4]}")

print("\n=== 4. INFERENCE: BORDERLINE UNCERTAINTY RECORDING (sz1.mat) ===")
with open("backend/data/aseeg/EO/sz/sz1.mat", "rb") as f:
    res = requests.post("http://127.0.0.1:8000/api/multimodal/analyze", files={"eeg_file": ("sz1.mat", f)}, data={"uncertainty_threshold": 0.65})
    unc_out = res.json()
    print(f"Prediction: {unc_out['predicted_label']} (Raw Label: {unc_out['raw_label']}) | Confidence: {unc_out['confidence']*100:.1f}%")
    print(f"Uncertain Flag: {unc_out['is_uncertain']} (Threshold: {unc_out['uncertainty_threshold']})")

print("\n=== 5. MODEL PERFORMANCE METRICS API ===")
ev = requests.get("http://127.0.0.1:8000/api/evaluation/latest/summary").json()
print(f"Overall Held-Out Test Accuracy: {ev['accuracy']*100:.1f}% (25 Participants)")
print(f"Accuracy on Accepted Cases: {ev['accuracy_on_accepted']*100:.1f}% (21 Participants)")
print(f"Prediction Coverage: {ev['coverage']*100:.1f}% (4 Cases Flagged as Uncertain)")
print(f"Unconstrained Confusion Matrix (Sum=25): {ev['confusion_matrix']}")
print(f"Gated Confusion Matrix (Sum=21): {ev['confusion_matrix_accepted']}")

print("\n=== 6. PERSISTED PREDICTION HISTORY ===")
hist = requests.get("http://127.0.0.1:8000/api/history?limit=3").json()
print(f"Total Persisted Records: {hist['total_count']}")
for r in hist["records"][:3]:
    print(f"  Record: {r['id']} | File: {r['filename']} | Pred: {r['predicted_label']} | Prob SZ: {r.get('prob_sz', 0)*100:.1f}% | Prob HC: {r.get('prob_hc', 0)*100:.1f}%")
