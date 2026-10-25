import pytest
import torch
import numpy as np
from app.config import WINDOW_SAMPLES
from app.models.multimodal_transformer import (
    ExplainableMultimodalTransformer,
    StructuralMRIEncoder,
    ClinicalEncoder,
    analyze_modality_agreement,
    MRI_ROI_FEATURES,
    CLINICAL_FEATURES
)

def test_multimodal_encoders_and_shapes():
    mri_enc = StructuralMRIEncoder(in_features=len(MRI_ROI_FEATURES), d_model=64)
    clin_enc = ClinicalEncoder(in_features=len(CLINICAL_FEATURES), d_model=64)

    dummy_mri = torch.randn(2, len(MRI_ROI_FEATURES))
    dummy_clin = torch.randn(2, len(CLINICAL_FEATURES))

    mri_out = mri_enc(dummy_mri)
    clin_out = clin_enc(dummy_clin)

    assert mri_out.shape == (2, 64)
    assert clin_out.shape == (2, 64)

def test_multimodal_transformer_forward_pass_all_modalities():
    model = ExplainableMultimodalTransformer(
        d_model=64,
        nhead=4,
        num_fusion_layers=2,
        eeg_num_channels=16
    )
    model.eval()

    # 4 clean EEG windows for a single subject
    eeg_windows = torch.randn(4, 16, WINDOW_SAMPLES)
    mri_feats = torch.randn(1, len(MRI_ROI_FEATURES))
    clin_feats = torch.randn(1, len(CLINICAL_FEATURES))

    out = model(eeg_windows=eeg_windows, mri_features=mri_feats, clinical_features=clin_feats)

    assert "fused_logits" in out
    assert out["fused_logits"].shape == (1, 2)
    assert set(out["available_modalities"]) == {"EEG", "MRI", "CLINICAL"}
    assert "eeg_logits" in out
    assert "mri_logits" in out
    assert "clinical_logits" in out

def test_multimodal_transformer_handles_missing_modalities():
    model = ExplainableMultimodalTransformer(d_model=64, nhead=4)
    model.eval()

    # Only EEG supplied
    eeg_windows = torch.randn(2, 16, WINDOW_SAMPLES)
    out_eeg = model(eeg_windows=eeg_windows, mri_features=None, clinical_features=None)
    assert out_eeg["available_modalities"] == ["EEG"]
    assert "eeg_logits" in out_eeg
    assert "mri_logits" not in out_eeg

    # Only Clinical supplied
    clin_feats = torch.randn(1, len(CLINICAL_FEATURES))
    out_clin = model(eeg_windows=None, mri_features=None, clinical_features=clin_feats)
    assert out_clin["available_modalities"] == ["CLINICAL"]

def test_modality_agreement_never_invents_agreement_for_single_modality():
    single_evidence = {
        "EEG": {"favored_diagnosis": "SCZ", "margin": 0.40, "prob_scz": 0.70, "prob_bd": 0.30}
    }
    status, score, desc = analyze_modality_agreement(single_evidence)
    assert status == "SINGLE_MODALITY"
    assert "Single" in desc or "requires at least two" in desc

def test_modality_agreement_detects_concordance_and_discordance():
    # Concordant: All support SCZ
    concordant = {
        "EEG": {"favored_diagnosis": "SCZ", "margin": 0.35, "prob_scz": 0.675, "prob_bd": 0.325},
        "MRI": {"favored_diagnosis": "SCZ", "margin": 0.45, "prob_scz": 0.725, "prob_bd": 0.275},
        "CLINICAL": {"favored_diagnosis": "SCZ", "margin": 0.25, "prob_scz": 0.625, "prob_bd": 0.375}
    }
    status, score, _ = analyze_modality_agreement(concordant)
    assert status == "FULL_AGREEMENT"
    assert score >= 0.70

    # Discordant: EEG favors SCZ, Clinical strongly favors BD
    discordant = {
        "EEG": {"favored_diagnosis": "SCZ", "margin": 0.50, "prob_scz": 0.75, "prob_bd": 0.25},
        "CLINICAL": {"favored_diagnosis": "BD", "margin": 0.60, "prob_scz": 0.20, "prob_bd": 0.80}
    }
    status_d, score_d, desc_d = analyze_modality_agreement(discordant)
    assert status_d == "DISCORDANT_DISAGREEMENT"
    assert "discordance" in desc_d.lower()

def test_differential_evidence_applies_discordance_penalty_and_uncertainty():
    model = ExplainableMultimodalTransformer(d_model=64, nhead=4)
    model.eval()

    eeg_windows = torch.randn(3, 16, WINDOW_SAMPLES)
    mri_feats = torch.randn(1, len(MRI_ROI_FEATURES))
    clin_feats = torch.randn(1, len(CLINICAL_FEATURES))

    res = model.evaluate_differential_evidence(
        eeg_windows=eeg_windows,
        mri_features=mri_feats,
        clinical_features=clin_feats,
        uncertainty_threshold=0.65
    )

    assert "predicted_label" in res
    assert "prob_scz" in res
    assert "prob_bd" in res
    assert abs((res["prob_scz"] + res["prob_bd"]) - 1.0) < 0.01
    assert res["predicted_label"] in ["SCZ", "BD", "UNCERTAIN"]
    assert "modality_agreement" in res
    assert "modality_evidence" in res
