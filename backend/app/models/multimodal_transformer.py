import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import WINDOW_SAMPLES, STANDARD_16_CHANNELS
from app.models.cnn_transformer import EEGCNNTransformer

# Canonical Structural MRI Regional Volumes & Cortical Thickness Markers
# Derived from literature: SCZ vs BD neuroimaging studies (ENIGMA & B-SNIP consortia)
MRI_ROI_FEATURES = [
    "left_lateral_ventricle",      # Typically SCZ >> BD > HC
    "right_lateral_ventricle",     # Typically SCZ >> BD > HC
    "left_hippocampus",            # SCZ volume reduction > BD
    "right_hippocampus",           # SCZ volume reduction > BD
    "left_amygdala",               # Often preserved/elevated in BD vs reduced in SCZ
    "right_amygdala",              # Often preserved/elevated in BD vs reduced in SCZ
    "left_thalamus",               # SCZ reduction > BD
    "right_thalamus",              # SCZ reduction > BD
    "superior_frontal_thickness",  # Cortical thinning in SCZ
    "rostral_middle_frontal_thickness",
    "inferior_temporal_thickness",
    "insula_cortical_thickness",
    "total_gray_matter_volume",
    "intracranial_volume_norm"
]

# Canonical Clinical Rating Scales for SCZ vs BD Differential Phenotyping
CLINICAL_FEATURES = [
    "panss_positive",       # Positive psychotic symptoms (delusions, hallucinations)
    "panss_negative",       # Negative symptoms (affective flattening, avolition - prominent in SCZ)
    "panss_general",        # General psychopathology
    "ymrs_score",           # Young Mania Rating Scale (hallmark of Bipolar manic/hypomanic states)
    "madrs_score",          # Montgomery-Asberg Depression Rating Scale (bipolar depressive phase)
    "age_of_onset",         # Typical SCZ (late teens/early 20s) vs BD (early-to-mid 20s)
    "duration_of_illness",  # In years
    "gaf_functioning"       # Global Assessment of Functioning (0-100)
]

class StructuralMRIEncoder(nn.Module):
    """
    Encoder for regional MRI morphometric and volumetric ROI features.
    Maps normalized MRI measurements into d_model embedding space.
    """
    def __init__(self, in_features: int = len(MRI_ROI_FEATURES), d_model: int = 64, dropout: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, in_features)
        return self.net(x)

class ClinicalEncoder(nn.Module):
    """
    Encoder for standardized neuropsychiatric clinical rating scales and demographics.
    Maps clinical vectors (PANSS, YMRS, MADRS, demographics) into d_model embedding space.
    """
    def __init__(self, in_features: int = len(CLINICAL_FEATURES), d_model: int = 64, dropout: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.net = nn.Sequential(
            nn.Linear(in_features, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, in_features)
        return self.net(x)

class MultimodalCrossAttentionFusion(nn.Module):
    """
    Bidirectional Cross-Attention Transformer Layer across available modality tokens.
    Learns temporal-spatial-phenotypic cross-modal dependencies.
    """
    def __init__(self, d_model: int = 64, nhead: int = 4, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, modality_tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        modality_tokens: (batch_size, num_modalities, d_model)
        key_padding_mask: (batch_size, num_modalities) boolean mask where True indicates missing modality
        """
        attn_out, _ = self.cross_attention(
            query=modality_tokens,
            key=modality_tokens,
            value=modality_tokens,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(modality_tokens + attn_out)
        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)
        return out

class ExplainableMultimodalTransformer(nn.Module):
    """
    Full Multimodal Transformer for SCZ-BD Differential Analysis.
    Integrates:
      1. Multi-channel EEG temporal-spatial encoder (CNN-Transformer)
      2. Structural MRI volumetric/morphometric encoder
      3. Neuropsychiatric clinical scale encoder
      4. Multimodal Cross-Modal Self-Attention Fusion
      5. Individual Modality Differential Evidence Heads
      6. Calibrated Fused Evidence Head
      7. Modality Concordance & Discordance Analysis
    """
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_fusion_layers: int = 2,
        eeg_num_channels: int = 16,
        mri_features: int = len(MRI_ROI_FEATURES),
        clinical_features: int = len(CLINICAL_FEATURES),
        dropout: float = 0.2
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Modality Encoders
        self.eeg_encoder = EEGCNNTransformer(
            num_channels=eeg_num_channels,
            time_points=WINDOW_SAMPLES,
            num_classes=2,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )
        self.mri_encoder = StructuralMRIEncoder(in_features=mri_features, d_model=d_model, dropout=dropout)
        self.clinical_encoder = ClinicalEncoder(in_features=clinical_features, d_model=d_model, dropout=dropout)

        # 2. Modality Type Embeddings (learned tokens to preserve modality identity)
        self.modality_type_embed = nn.Embedding(3, d_model) # 0: EEG, 1: MRI, 2: Clinical

        # 3. Multimodal Cross-Attention Layers
        self.fusion_layers = nn.ModuleList([
            MultimodalCrossAttentionFusion(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(num_fusion_layers)
        ])

        # 4. Individual Modality Evidence Heads (Outputs 2 logits: [SCZ, BD])
        self.eeg_head = nn.Linear(d_model, 2)
        self.mri_head = nn.Linear(d_model, 2)
        self.clinical_head = nn.Linear(d_model, 2)

        # 5. Multimodal Fused Classifier Head
        self.fused_classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(
        self,
        eeg_windows: Optional[torch.Tensor] = None,
        mri_features: Optional[torch.Tensor] = None,
        clinical_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass accommodating any combination of available modalities.
        Returns logits for fused prediction and for individual modalities.
        """
        tokens = []
        masks = []
        modality_keys = []

        device = next(self.parameters()).device

        # Extract EEG representation
        if eeg_windows is not None and eeg_windows.numel() > 0:
            if eeg_windows.dim() == 4:
                # Batched subjects: (batch_size, num_windows, channels, times)
                batch_size = eeg_windows.size(0)
                eeg_reps = []
                for b in range(batch_size):
                    rep, _ = self.eeg_encoder.subject_pooler(self.eeg_encoder.extract_window_features(eeg_windows[b]))
                    eeg_reps.append(rep)
                eeg_emb = torch.cat(eeg_reps, dim=0) # (B, d_model)
            else:
                # Single subject windows: (num_windows, channels, times)
                w_feats = self.eeg_encoder.extract_window_features(eeg_windows)
                eeg_emb, _ = self.eeg_encoder.subject_pooler(w_feats) # (1, d_model)
            eeg_token = eeg_emb + self.modality_type_embed(torch.tensor(0, device=device))
            tokens.append(eeg_token.unsqueeze(1)) # (B, 1, d_model)
            masks.append(False)
            modality_keys.append("EEG")
        else:
            eeg_token = None

        # Extract MRI representation
        if mri_features is not None and mri_features.numel() > 0:
            if mri_features.dim() == 1:
                mri_features = mri_features.unsqueeze(0)
            mri_emb = self.mri_encoder(mri_features)
            mri_token = mri_emb + self.modality_type_embed(torch.tensor(1, device=device))
            tokens.append(mri_token.unsqueeze(1))
            masks.append(False)
            modality_keys.append("MRI")
        else:
            mri_token = None

        # Extract Clinical representation
        if clinical_features is not None and clinical_features.numel() > 0:
            if clinical_features.dim() == 1:
                clinical_features = clinical_features.unsqueeze(0)
            clin_emb = self.clinical_encoder(clinical_features)
            clin_token = clin_emb + self.modality_type_embed(torch.tensor(2, device=device))
            tokens.append(clin_token.unsqueeze(1))
            masks.append(False)
            modality_keys.append("CLINICAL")
        else:
            clin_token = None

        if len(tokens) == 0:
            raise ValueError("At least one modality must be provided to the Multimodal Transformer.")

        # Stack available modality tokens: (batch_size, num_available_modalities, d_model)
        seq_tokens = torch.cat(tokens, dim=1)

        # Cross-Modal Attention Fusion
        fused_seq = seq_tokens
        for layer in self.fusion_layers:
            fused_seq = layer(fused_seq)

        # Fused token is pooled across available modalities
        fused_rep = torch.mean(fused_seq, dim=1) # (batch_size, d_model)
        fused_logits = self.fused_classifier(fused_rep)

        out: Dict[str, Any] = {
            "fused_logits": fused_logits,
            "available_modalities": modality_keys
        }

        # Modality-specific evidence logits
        if eeg_token is not None:
            out["eeg_logits"] = self.eeg_head(eeg_token)
        if mri_token is not None:
            out["mri_logits"] = self.mri_head(mri_token)
        if clin_token is not None:
            out["clinical_logits"] = self.clinical_head(clin_token)

        return out

    def evaluate_differential_evidence(
        self,
        eeg_windows: Optional[torch.Tensor] = None,
        mri_features: Optional[torch.Tensor] = None,
        clinical_features: Optional[torch.Tensor] = None,
        uncertainty_threshold: float = 0.65,
        target_task: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes inference, computes separate evidence for each modality and the fused model,
        analyzes modality concordance/discordance, and applies uncertainty guardrails.
        Explicitly separates SZ vs HC technical validation from SCZ vs BD differential tasks.
        """
        self.eval()
        is_sz_hc = target_task in ["SZ_VS_HC", "SZ_VS_HC_TECH_VAL"]
        lbl_0 = "SZ" if is_sz_hc else "SCZ"
        lbl_1 = "HC" if is_sz_hc else "BD"

        with torch.no_grad():
            res = self.forward(eeg_windows, mri_features, clinical_features)
            available = res["available_modalities"]

            # Fused Evidence
            fused_probs = F.softmax(res["fused_logits"], dim=-1).squeeze(0).cpu().numpy()
            p_scz_fused = float(fused_probs[0])
            p_bd_fused = float(fused_probs[1]) if len(fused_probs) > 1 else float(1.0 - p_scz_fused)

            # Modality-Specific Evidences
            modality_evidence = {}
            if "eeg_logits" in res:
                eeg_p = F.softmax(res["eeg_logits"], dim=-1).squeeze(0).cpu().numpy()
                modality_evidence["EEG"] = {
                    "prob_scz": round(float(eeg_p[0]), 4),
                    "prob_bd": round(float(eeg_p[1]), 4),
                    "prob_sz": round(float(eeg_p[0]), 4),
                    "prob_hc": round(float(eeg_p[1]), 4),
                    "favored_diagnosis": lbl_0 if eeg_p[0] >= eeg_p[1] else lbl_1,
                    "margin": round(abs(float(eeg_p[0] - eeg_p[1])), 4)
                }

            if "mri_logits" in res:
                mri_p = F.softmax(res["mri_logits"], dim=-1).squeeze(0).cpu().numpy()
                modality_evidence["MRI"] = {
                    "prob_scz": round(float(mri_p[0]), 4),
                    "prob_bd": round(float(mri_p[1]), 4),
                    "favored_diagnosis": lbl_0 if mri_p[0] >= mri_p[1] else lbl_1,
                    "margin": round(abs(float(mri_p[0] - mri_p[1])), 4)
                }

            if "clinical_logits" in res:
                clin_p = F.softmax(res["clinical_logits"], dim=-1).squeeze(0).cpu().numpy()
                modality_evidence["CLINICAL"] = {
                    "prob_scz": round(float(clin_p[0]), 4),
                    "prob_bd": round(float(clin_p[1]), 4),
                    "favored_diagnosis": lbl_0 if clin_p[0] >= clin_p[1] else lbl_1,
                    "margin": round(abs(float(clin_p[0] - clin_p[1])), 4)
                }

            # Modality Agreement / Discordance Analysis
            agreement_status, agreement_score, agreement_desc = analyze_modality_agreement(modality_evidence)

            # Apply discordance penalty to confidence if modalities disagree
            raw_confidence = max(p_scz_fused, p_bd_fused)
            calibrated_confidence = raw_confidence
            discordance_penalty = 0.0

            if agreement_status == "DISCORDANT_DISAGREEMENT":
                discordance_penalty = round((1.0 - agreement_score) * 0.20, 4)
                calibrated_confidence = max(0.50, raw_confidence - discordance_penalty)

            # Uncertainty check against threshold
            is_uncertain = calibrated_confidence < uncertainty_threshold
            pred_class = lbl_0 if p_scz_fused >= p_bd_fused else lbl_1
            final_label = "UNCERTAIN" if is_uncertain else pred_class

            return {
                "predicted_label": final_label,
                "raw_label": pred_class,
                "target_task": target_task or ("SZ_VS_HC" if is_sz_hc else "SCZ_VS_BD"),
                "prob_scz": round(p_scz_fused, 4),
                "prob_bd": round(p_bd_fused, 4),
                "prob_sz": round(p_scz_fused, 4),
                "prob_hc": round(p_bd_fused, 4),
                "confidence": round(calibrated_confidence, 4),
                "raw_confidence": round(raw_confidence, 4),
                "uncertainty_threshold": uncertainty_threshold,
                "is_uncertain": is_uncertain,
                "discordance_penalty": discordance_penalty,
                "evidence_margin": round(abs(p_scz_fused - p_bd_fused), 4),
                "available_modalities": available,
                "missing_modalities": [m for m in ["EEG", "MRI", "CLINICAL"] if m not in available],
                "modality_evidence": modality_evidence,
                "modality_agreement": {
                    "status": agreement_status,
                    "concordance_score": round(agreement_score, 4),
                    "description": agreement_desc
                }
            }

def analyze_modality_agreement(modality_evidence: Dict[str, Dict[str, Any]]) -> Tuple[str, float, str]:
    """
    Evaluates directional agreement and cosine concordance across supported modalities.
    Distinguishes:
      - FULL_AGREEMENT: All modalities support the same diagnostic direction.
      - PARTIAL_AGREEMENT: Majority support one direction with weak or neutral secondary modality.
      - DISCORDANT_DISAGREEMENT: Modalities strongly favor opposite diagnoses.
      - SINGLE_MODALITY: Only one modality supplied (never invents agreement).
    """
    n_mods = len(modality_evidence)
    if n_mods <= 1:
        mod_name = list(modality_evidence.keys())[0] if n_mods == 1 else "None"
        return (
            "SINGLE_MODALITY",
            1.0,
            f"Only {mod_name} modality supplied. Multi-modality concordance analysis requires at least two distinct modalities."
        )

    favored = [m["favored_diagnosis"] for m in modality_evidence.values()]
    scz_count = favored.count("SCZ")
    bd_count = favored.count("BD")

    margins = [m["margin"] for m in modality_evidence.values()]
    avg_margin = float(np.mean(margins))

    if scz_count == n_mods or bd_count == n_mods:
        status = "FULL_AGREEMENT"
        score = min(1.0, 0.70 + 0.30 * avg_margin)
        diag = "SCZ" if scz_count == n_mods else "BD"
        desc = f"All {n_mods} evaluated modalities consistently support the {diag} hypothesis."
    elif n_mods >= 3 and (scz_count >= 2 or bd_count >= 2):
        status = "PARTIAL_AGREEMENT"
        majority = "SCZ" if scz_count >= 2 else "BD"
        score = 0.65
        desc = f"Majority of modalities ({max(scz_count, bd_count)}/{n_mods}) support {majority}, with minority divergence."
    else:
        status = "DISCORDANT_DISAGREEMENT"
        score = max(0.20, 0.50 - avg_margin * 0.5)
        desc = (
            "Cross-modal discordance detected: Electro-oscillatory and phenotypic/structural evidence "
            "point in opposing diagnostic directions. Heightened uncertainty penalty enforced."
        )

    return status, score, desc
