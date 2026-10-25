import torch
import pytest
from app.models.cnn_transformer import EEGCNNTransformer

def test_uncertainty_threshold_rejects_low_confidence():
    model = EEGCNNTransformer(
        num_channels=16,
        time_points=500,
        num_classes=2,
        d_model=32
    )
    
    windows = torch.randn(5, 16, 500)

    # Test with very high threshold (e.g. 0.99) -> must flag UNCERTAIN
    res_high_thresh = model.predict_with_uncertainty(windows, uncertainty_threshold=0.99)
    assert res_high_thresh["is_uncertain"] is True
    assert res_high_thresh["predicted_label"] == "UNCERTAIN"
    assert res_high_thresh["confidence"] < 0.99

    # Test with baseline threshold (e.g. 0.50) -> should assign raw label
    res_low_thresh = model.predict_with_uncertainty(windows, uncertainty_threshold=0.50)
    assert res_low_thresh["is_uncertain"] is False
    assert res_low_thresh["predicted_label"] in ["SCZ", "BD"]
