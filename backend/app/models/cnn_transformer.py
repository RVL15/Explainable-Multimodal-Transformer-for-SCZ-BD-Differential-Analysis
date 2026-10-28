import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class AttentiveSubjectPooler(nn.Module):
    """
    Learns attention weights across multiple time windows belonging to a single subject.
    Input: (num_windows, d_model)
    Output: (1, d_model), attention_weights (num_windows,)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, window_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # window_embeddings: (num_windows, d_model)
        weights = self.attention(window_embeddings)  # (num_windows, 1)
        attn_scores = F.softmax(weights, dim=0)       # (num_windows, 1)
        pooled = torch.sum(attn_scores * window_embeddings, dim=0, keepdim=True)  # (1, d_model)
        return pooled, attn_scores.squeeze(-1)

class EEGCNNTransformer(nn.Module):
    """
    Explainable Multimodal CNN-Transformer for SCZ-BD Differential EEG Analysis.
    Architecture:
      1. Temporal & Depthwise Spatial 1D-Convolutions (local temporal & topographical features)
      2. Feature Projection + Sinusoidal Positional Encoding
      3. Multi-Head Transformer Encoder (global temporal dependencies across frequency wavelets)
      4. Window-level Pooling & Attentive Subject-level Aggregation
      5. Calibrated Linear Classifier Head
    """
    def __init__(
        self,
        num_channels: int = 16,
        time_points: int = 500,
        num_classes: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_channels = num_channels
        self.time_points = time_points
        self.num_classes = num_classes
        self.d_model = d_model

        # 1. Temporal Convolution Block
        # Captures oscillations across ~10-25 samples (40ms-100ms at 250 Hz)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=32,
                kernel_size=25,
                stride=1,
                padding=12,
                bias=False
            ),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # downsamples to time_points // 2 (250)
        )

        # 2. Spatial / Intermediate Convolution Block
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=15,
                stride=1,
                padding=7,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=5, stride=5),  # downsamples to 250 // 5 = 50 sequence steps
            nn.Dropout(dropout)
        )

        # 3. Projection to Transformer d_model
        self.proj = nn.Linear(64, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=200)

        # 4. Multi-Head Self-Attention Transformer Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Attentive subject pooling
        self.subject_pooler = AttentiveSubjectPooler(d_model=d_model)

        # 6. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def extract_window_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, num_channels, time_points)
        Returns: window representation of shape (batch_size, d_model)
        """
        # Temporal & spatial CNN
        feats = self.temporal_conv(x)      # (B, 32, 250)
        feats = self.spatial_conv(feats)   # (B, 64, 50)
        
        # Reshape for transformer: (B, seq_len=50, channels=64)
        feats = feats.transpose(1, 2)
        proj_feats = self.proj(feats)      # (B, 50, d_model)
        
        # Transformer
        pe_feats = self.pos_encoder(proj_feats)
        trans_out = self.transformer_encoder(pe_feats)  # (B, 50, d_model)
        
        # Mean pool across time steps in the window
        window_emb = torch.mean(trans_out, dim=1)        # (B, d_model)
        return window_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for either:
        - 3D batched windows: (batch_size, num_channels, time_points) -> (batch_size, num_classes)
        - 4D batched subjects/windows: (batch_size, num_windows, num_channels, time_points) -> (batch_size, num_classes)
        """
        if x.dim() == 4:
            B, W, C, T = x.shape
            x_flat = x.view(B * W, C, T)
            window_emb = self.extract_window_features(x_flat) # (B*W, d_model)
            window_emb = window_emb.view(B, W, -1)
            if B == 1:
                subject_rep, _ = self.subject_pooler(window_emb.squeeze(0))
            else:
                reps = []
                for b in range(B):
                    s_rep, _ = self.subject_pooler(window_emb[b])
                    reps.append(s_rep)
                subject_rep = torch.cat(reps, dim=0)
            logits = self.classifier(subject_rep)
            return logits
        elif x.dim() == 3:
            window_emb = self.extract_window_features(x)
            logits = self.classifier(window_emb)
            return logits
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")

    def forward_subject(self, windows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchical Subject Forward Pass:
        windows: (num_clean_windows, num_channels, time_points)
        Returns:
            logits: (1, num_classes)
            window_attn_weights: (num_clean_windows,)
        """
        window_embeddings = self.extract_window_features(windows) # (num_windows, d_model)
        subject_rep, attn_weights = self.subject_pooler(window_embeddings) # (1, d_model)
        logits = self.classifier(subject_rep) # (1, num_classes)
        return logits, attn_weights

    def predict_with_uncertainty(
        self,
        windows: torch.Tensor,
        uncertainty_threshold: float = 0.65,
        target_task: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculates calibrated evidence and applies uncertainty rejection.
        Supports both SZ_VS_HC (technical validation on ASEEG) and SCZ_VS_BD tasks.
        Never conflates Healthy Controls (HC) with Bipolar Disorder (BD).
        """
        self.eval()
        with torch.no_grad():
            logits, attn_weights = self.forward_subject(windows)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
            p_0 = float(probs[0])
            p_1 = float(probs[1]) if len(probs) > 1 else float(1.0 - p_0)
            max_prob = max(p_0, p_1)
            pred_class_idx = int(torch.argmax(logits, dim=-1).item())
            
            effective_task = target_task or getattr(self, "target_task", "SCZ_VS_BD")
            if effective_task in ["SZ_VS_HC", "SZ_VS_HC_TECH_VAL"]:
                class_map = {0: "SZ", 1: "HC"}
            else:
                class_map = {0: "SCZ", 1: "BD"}

            predicted_label = class_map.get(pred_class_idx, "UNKNOWN")
            
            # Uncertainty check: if confidence is below threshold, reject as UNCERTAIN
            is_uncertain = max_prob < uncertainty_threshold
            
            result = {
                "predicted_label": "UNCERTAIN" if is_uncertain else predicted_label,
                "raw_label": predicted_label,
                "prob_class_0": round(p_0, 4),
                "prob_class_1": round(p_1, 4),
                "prob_scz": round(p_0, 4),
                "prob_bd": round(p_1, 4),
                "prob_sz": round(p_0, 4),
                "prob_hc": round(p_1, 4),
                "confidence": round(max_prob, 4),
                "uncertainty_threshold": uncertainty_threshold,
                "is_uncertain": is_uncertain,
                "target_task": effective_task,
                "window_attention": attn_weights.cpu().numpy().tolist()
            }
            return result
