import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from app.config import STANDARD_16_CHANNELS, WINDOW_SAMPLES
from app.models.cnn_transformer import EEGCNNTransformer

SCIENTIFIC_LIMITATION_DISCLAIMER = (
    "RESEARCH LIMITATION NOTICE: Gradient-based feature attributions (Integrated Gradients / "
    "Input x Gradient) reflect mathematical sensitivity of the neural network's trained decision surface. "
    "Attribution saliency does NOT imply neurobiological causation, verified physiological biomarkers, "
    "or clinical etiology. This prototype is for engineering and scientific evaluation only."
)

def compute_gradient_attribution(
    model: EEGCNNTransformer,
    eeg_windows: torch.Tensor,
    target_class: int = 0,
    steps: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Integrated Gradients attribution for EEG tensor input.
    Input: eeg_windows of shape (N, num_channels, time_points)
    Returns:
        attributions: np.ndarray of shape (N, num_channels, time_points)
        channel_importance: np.ndarray of shape (num_channels,)
    """
    model.eval()
    
    # Try using Captum IntegratedGradients if available, or fall back to native Gauss-Legendre IG
    try:
        from captum.attr import IntegratedGradients
        ig = IntegratedGradients(model)
        # Baseline is zero tensor
        baseline = torch.zeros_like(eeg_windows)
        attributions_tensor = ig.attribute(
            eeg_windows,
            baselines=baseline,
            target=target_class,
            n_steps=steps
        )
        attr = attributions_tensor.detach().cpu().numpy()
    except Exception:
        # Native robust Integrated Gradients implementation
        baseline = torch.zeros_like(eeg_windows)
        total_grads = torch.zeros_like(eeg_windows)
        
        for alpha in np.linspace(0.0, 1.0, steps):
            interpolated = (baseline + alpha * (eeg_windows - baseline)).clone().detach().requires_grad_(True)
            logits = model(interpolated)
            target_logit = logits[:, target_class].sum()
            model.zero_grad()
            target_logit.backward()
            total_grads += interpolated.grad.detach()

        avg_grads = total_grads / steps
        attr = ((eeg_windows - baseline) * avg_grads).detach().cpu().numpy()

    # Aggregate across windows and time points to get channel importance:
    # Mean of absolute gradient attribution
    channel_importance = np.mean(np.abs(attr), axis=(0, 2))
    total_imp = np.sum(channel_importance)
    if total_imp > 0:
        channel_importance = channel_importance / total_imp

    return attr, channel_importance

def explain_subject_prediction(
    model: EEGCNNTransformer,
    clean_windows: np.ndarray,
    target_class: Optional[int] = None,
    channel_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generates complete explainability package for a subject's EEG recording:
    - Channel ranking
    - Window/time segment contribution
    - Pointwise attribution overlays
    - Explicit scientific limitations
    """
    if channel_names is None:
        channel_names = STANDARD_16_CHANNELS[:clean_windows.shape[1]]

    # Convert to torch tensor
    x_tensor = torch.from_numpy(clean_windows[:16, :16, :WINDOW_SAMPLES]).float()
    
    # Determine predicted class if target_class is not specified
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor)
        mean_logits = torch.mean(logits, dim=0)
        pred_class = int(torch.argmax(mean_logits).item())
        probs = torch.softmax(mean_logits, dim=-1).cpu().numpy()

    target_class_to_explain = pred_class if target_class is None else target_class
    
    # Compute attributions
    attr, ch_importance = compute_gradient_attribution(
        model=model,
        eeg_windows=x_tensor,
        target_class=target_class_to_explain,
        steps=25
    )

    # 1. Channel Attributions
    channel_items = []
    ranked_indices = np.argsort(-ch_importance)
    for rank, idx in enumerate(ranked_indices):
        ch_name = channel_names[idx] if idx < len(channel_names) else f"Ch_{idx+1}"
        channel_items.append({
            "channel": ch_name,
            "importance_score": round(float(ch_importance[idx]), 4),
            "rank": rank + 1
        })

    # 2. Time Window Attributions
    # Mean attribution for each window
    window_importance = np.mean(np.abs(attr), axis=(1, 2))
    w_sum = np.sum(window_importance)
    if w_sum > 0:
        window_importance = window_importance / w_sum

    time_items = []
    for w_idx, w_imp in enumerate(window_importance):
        time_items.append({
            "window_idx": w_idx,
            "start_sec": round(w_idx * 2.0, 1),
            "end_sec": round((w_idx + 1) * 2.0, 1),
            "attribution_weight": round(float(w_imp), 4)
        })

    top_channels = [c["channel"] for c in channel_items[:4]]

    return {
        "target_class_explained": "SCZ" if target_class_to_explain == 0 else "BD",
        "predicted_class": "SCZ" if pred_class == 0 else "BD",
        "prob_scz": round(float(probs[0]), 4),
        "prob_bd": round(float(probs[1]), 4) if len(probs) > 1 else round(float(1.0 - probs[0]), 4),
        "channel_attributions": channel_items,
        "time_attributions": time_items,
        "top_contributing_channels": top_channels,
        "method": "Integrated Gradients (Path-Integral Feature Attribution)",
        "scientific_disclaimer": SCIENTIFIC_LIMITATION_DISCLAIMER
    }
