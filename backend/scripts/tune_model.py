import sys
sys.path.append("backend")
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from app.database import get_db
from app.models.pipeline import get_subject_level_partitions, prepare_partition_dataset, preprocess_and_window_eeg
from app.models.cnn_transformer import EEGCNNTransformer
from app.eeg.io import load_eeg_file
from app.evaluation.metrics import compute_classification_metrics

with get_db() as conn:
    c = conn.cursor()
    c.execute("SELECT subject_id, diagnostic_label, recording_path FROM subjects WHERE dataset_id='aseeg' AND is_excluded=0 AND file_exists=1")
    rows = [dict(r) for r in c.fetchall()]

train_subs, val_subs, test_subs = get_subject_level_partitions(rows, test_ratio=0.25, val_ratio=0.15, random_seed=42)

X_train, y_train, _, norm_stats = prepare_partition_dataset(train_subs, "SZ_VS_HC")
X_val, y_val, _, _ = prepare_partition_dataset(val_subs, "SZ_VS_HC", norm_stats=norm_stats)

for lr in [5e-4, 3e-4]:
    for seed in [42, 123]:
        torch.manual_seed(seed)
        model = EEGCNNTransformer(num_channels=16, time_points=500, num_classes=2, d_model=64, dropout=0.25)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=32, shuffle=False)

        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, 9):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    preds = model(bx).argmax(dim=-1)
                    correct += int((preds == by).sum().item())
            val_acc = correct / len(X_val)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Evaluate best state on test subjects at subject level
        if best_state is None:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        test_model = EEGCNNTransformer(num_channels=16, time_points=500, num_classes=2, d_model=64)
        test_model.load_state_dict(best_state)
        test_model.eval()

        y_true = []
        y_pred = []
        y_scores = []
        for sub in test_subs:
            sig, chs, fs = load_eeg_file(sub["recording_path"])
            windows = preprocess_and_window_eeg(sig, chs, fs)
            norm_w = (windows - norm_stats[0]) / norm_stats[1]
            tensor_w = torch.from_numpy(norm_w).float()
            res = test_model.predict_with_uncertainty(tensor_w, uncertainty_threshold=0.5, target_task="SZ_VS_HC")
            true_idx = 0 if sub["diagnostic_label"] == "SZ" else 1
            pred_idx = 0 if res["raw_label"] == "SZ" else 1
            y_true.append(true_idx)
            y_pred.append(pred_idx)
            y_scores.append(res["prob_hc"])

        m = compute_classification_metrics(y_true, y_pred, y_scores)
        print(f"lr={lr}, seed={seed} | Best Val Acc: {best_val_acc:.4f} | Test Acc: {m['accuracy']:.4f}, F1: {m['f1_score']:.4f}, AUC: {m['roc_auc']}, Matrix: {m['confusion_matrix']}")
