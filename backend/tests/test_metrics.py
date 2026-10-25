import pytest
from app.evaluation.metrics import compute_classification_metrics

def test_genuine_metrics_formulae():
    # 4 True SCZ (0), 4 True BD (1)
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    # Predictions:
    # True positives for class 1 (BD): 3
    # False negative: 1 (predicted 0)
    # True negatives: 3
    # False positive: 1 (predicted 1)
    y_pred = [0, 0, 0, 1, 1, 1, 1, 0]
    y_scores = [0.1, 0.2, 0.3, 0.8, 0.9, 0.85, 0.75, 0.4]

    metrics = compute_classification_metrics(y_true, y_pred, y_scores)

    # TP = 3, TN = 3, FP = 1, FN = 1
    # Accuracy = (3+3)/8 = 0.75
    # Precision = 3/(3+1) = 0.75
    # Recall / Sens = 3/(3+1) = 0.75
    # Spec = 3/(3+1) = 0.75
    # F1 = 0.75
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.75
    assert metrics["f1_score"] == 0.75
    assert metrics["sensitivity"] == 0.75
    assert metrics["specificity"] == 0.75
    assert metrics["confusion_matrix"] == [[3, 1], [1, 3]]
    assert metrics["uncertain_count"] == 0
    assert metrics["coverage"] == 1.0

def test_metrics_handles_uncertainty_coverage():
    # 10 subjects, 2 uncertain (-1)
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, -1, 1, 1, 1, 1, -1]
    
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["uncertain_count"] == 2
    assert metrics["coverage"] == 0.8  # 8 / 10
    # Decided 8 subjects are all correctly classified
    assert metrics["accuracy"] == 1.0
