import torch
import pytest
from app.models.cnn_transformer import EEGCNNTransformer

def test_cnn_transformer_shapes_and_forward_pass():
    batch_size = 4
    num_channels = 16
    time_points = 500  # 2 seconds at 250 Hz
    num_classes = 2

    model = EEGCNNTransformer(
        num_channels=num_channels,
        time_points=time_points,
        num_classes=num_classes,
        d_model=64,
        nhead=4,
        num_layers=2
    )

    # Input tensor
    x = torch.randn(batch_size, num_channels, time_points)
    
    # 1. Forward pass on batched windows
    logits = model(x)
    assert logits.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN detected in model output logits"

    # 2. Subject forward pass across multiple windows
    num_clean_windows = 8
    subject_windows = torch.randn(num_clean_windows, num_channels, time_points)
    sub_logits, attn_weights = model.forward_subject(subject_windows)
    
    assert sub_logits.shape == (1, num_classes)
    assert attn_weights.shape == (num_clean_windows,)
    # Attention weights must sum to 1.0
    assert torch.isclose(attn_weights.sum(), torch.tensor(1.0), atol=1e-4)
