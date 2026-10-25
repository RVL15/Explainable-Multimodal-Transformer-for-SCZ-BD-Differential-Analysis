import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from app.eeg.spectral import extract_subject_spectral_features

class BaselineLogisticRegression:
    """
    Interpretable classical machine learning baseline.
    Trained on documented neurophysiological spectral features:
    - Channel-wise relative band powers (Delta, Theta, Alpha, Beta, Gamma)
    - Theta / Alpha ratio
    - Frontal Alpha Asymmetry index
    """
    def __init__(self, c_reg: float = 1.0, random_state: int = 42):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=c_reg, max_iter=500, random_state=random_state, class_weight="balanced"))
        ])
        self.is_fitted = False
        self.classes_ = ["SCZ", "BD"]

    def fit(self, X_features: np.ndarray, y_labels: np.ndarray):
        """
        X_features: (num_subjects, num_features)
        y_labels: 1D array of integers (0: SCZ, 1: BD)
        """
        self.pipeline.fit(X_features, y_labels)
        self.is_fitted = True

    def predict_proba(self, X_features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Baseline model has not been trained yet.")
        return self.pipeline.predict_proba(X_features)

    def predict_subject(
        self,
        raw_eeg_signal: np.ndarray,
        sample_rate: float,
        uncertainty_threshold: float = 0.65
    ) -> Dict[str, Any]:
        """
        Extracts spectral features from subject EEG and predicts diagnosis.
        """
        feat_vec = extract_subject_spectral_features(raw_eeg_signal, sample_rate).reshape(1, -1)
        probs = self.predict_proba(feat_vec)[0]
        
        p_scz = float(probs[0])
        p_bd = float(probs[1]) if len(probs) > 1 else float(1.0 - p_scz)
        max_prob = max(p_scz, p_bd)
        pred_idx = int(np.argmax(probs))
        pred_label = self.classes_[pred_idx]
        
        is_uncertain = bool(max_prob < uncertainty_threshold)
        
        return {
            "predicted_label": "UNCERTAIN" if is_uncertain else pred_label,
            "raw_label": pred_label,
            "prob_scz": round(p_scz, 4),
            "prob_bd": round(p_bd, 4),
            "confidence": round(max_prob, 4),
            "is_uncertain": is_uncertain,
            "features_extracted_count": feat_vec.shape[1]
        }

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "BaselineLogisticRegression":
        return joblib.load(filepath)
