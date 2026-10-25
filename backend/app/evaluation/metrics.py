import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import roc_auc_score

def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_scores: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Computes genuine scientific metrics from real evaluated labels and predictions.
    Handles uncertain predictions (marked as -1) separately with coverage calculation.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    total_subjects = len(y_true_arr)

    if total_subjects == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "roc_auc": None,
            "uncertain_count": 0,
            "coverage": 1.0,
            "confusion_matrix": [[0, 0], [0, 0]]
        }

    # Identify uncertain predictions
    uncertain_mask = (y_pred_arr == -1)
    uncertain_count = int(np.sum(uncertain_mask))
    coverage = float((total_subjects - uncertain_count) / total_subjects)

    # Calculate metrics on the covered (decided) subset
    valid_mask = ~uncertain_mask
    if np.sum(valid_mask) > 0:
        val_true = y_true_arr[valid_mask]
        val_pred = y_pred_arr[valid_mask]

        # Binary confusion matrix (0: Negative/SCZ, 1: Positive/BD)
        tp = int(np.sum((val_true == 1) & (val_pred == 1)))
        tn = int(np.sum((val_true == 0) & (val_pred == 0)))
        fp = int(np.sum((val_true == 0) & (val_pred == 1)))
        fn = int(np.sum((val_true == 1) & (val_pred == 0)))

        acc = float((tp + tn) / len(val_true)) if len(val_true) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        sens = rec
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    else:
        tp, tn, fp, fn = 0, 0, 0, 0
        acc, prec, rec, sens, spec, f1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # ROC-AUC calculation if continuous scores provided
    roc_auc = None
    if y_scores is not None and len(y_scores) == total_subjects:
        try:
            # Requires at least one positive and one negative class
            if len(np.unique(y_true_arr)) > 1:
                roc_auc = float(roc_auc_score(y_true_arr, y_scores))
                roc_auc = round(roc_auc, 4)
        except Exception:
            roc_auc = None

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "roc_auc": roc_auc,
        "uncertain_count": uncertain_count,
        "coverage": round(coverage, 4),
        "confusion_matrix": [
            [tn, fp],
            [fn, tp]
        ]
    }
