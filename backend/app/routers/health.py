import os
import csv
from typing import Dict, Any, Optional
from fastapi import APIRouter
import torch
import mne
import numpy as np
import scipy
import sklearn

from app.database import get_db
from app.config import CHECKPOINTS_DIR, MANIFESTS_DIR, WINDOW_SAMPLES
from app.eeg.validation import get_supported_formats_info
from app.models.cnn_transformer import EEGCNNTransformer

router = APIRouter(prefix="/api/health", tags=["Health"])

def compute_tasks_readiness() -> Dict[str, Any]:
    """
    Dynamically verifies dataset availability, manifest validity, checkpoint existence,
    and model inference readiness for both classification tasks.
    """
    # 1. Schizophrenia vs. Healthy Controls (SZ_VS_HC)
    manifest_path = MANIFESTS_DIR / "aseeg_manifest.csv"
    aseeg_participants_count = None
    aseeg_sampling_rate = None
    aseeg_channels = None
    aseeg_manifest_verified = False

    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                aseeg_participants_count = len(rows)
                if rows:
                    aseeg_sampling_rate = float(rows[0].get("sampling_rate", 500.0))
                    aseeg_channels = int(rows[0].get("channels_count", 19))
                aseeg_manifest_verified = (aseeg_participants_count == 97)
        except Exception:
            pass

    best_cp = CHECKPOINTS_DIR / "best_model.pt"
    sz_hc_status = "STATUS UNKNOWN"
    sz_hc_status_reason = ""
    sz_hc_arch = None
    sz_hc_active_checkpoint = None

    if best_cp.exists():
        try:
            state = torch.load(str(best_cp), map_location="cpu", weights_only=False)
            if "model_state_dict" in state:
                # Confirm architecture and test real forward pass
                model = EEGCNNTransformer(num_channels=16, time_points=WINDOW_SAMPLES, num_classes=2)
                model.load_state_dict(state["model_state_dict"])
                model.eval()
                dummy_input = torch.zeros(1, 16, WINDOW_SAMPLES, dtype=torch.float32)
                with torch.no_grad():
                    out = model(dummy_input)
                if out.shape == (1, 2):
                    sz_hc_status = "MODEL READY"
                    sz_hc_arch = "CNN–Transformer"
                    sz_hc_active_checkpoint = "best_model.pt"
                else:
                    sz_hc_status = "STATUS UNKNOWN"
                    sz_hc_status_reason = f"Model forward pass returned shape {out.shape} instead of (1, 2)."
            else:
                sz_hc_status = "MODEL NOT TRAINED"
                sz_hc_status_reason = "Checkpoint missing model_state_dict."
        except Exception as e:
            sz_hc_status = "STATUS UNKNOWN"
            sz_hc_status_reason = f"Failed to load checkpoint: {str(e)}"
    else:
        sz_hc_status = "MODEL NOT TRAINED"
        sz_hc_status_reason = "best_model.pt checkpoint does not exist."

    # Held-out test accuracy from saved evaluation results
    sz_hc_accuracy = None
    report_path = CHECKPOINTS_DIR.parent / "reports" / "final_evaluation_report.json"
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                rep_data = json.load(f)
                acc = rep_data.get("unconstrained_evaluation", {}).get("accuracy")
                if acc is not None:
                    sz_hc_accuracy = round(float(acc) * 100, 1)
        except Exception:
            pass

    if sz_hc_accuracy is None:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT ev.accuracy
            FROM evaluations ev
            JOIN experiments e ON ev.experiment_id = e.id
            WHERE (e.id = 'exp_real_4cbbe6c5' OR e.target_task = 'SZ_VS_HC' OR e.dataset_id = 'aseeg')
              AND e.status = 'COMPLETED'
            ORDER BY CASE WHEN e.id = 'exp_real_4cbbe6c5' THEN 0 ELSE 1 END, ev.created_at DESC LIMIT 1
            """)
            row = cursor.fetchone()
            if row and row["accuracy"] is not None:
                sz_hc_accuracy = round(float(row["accuracy"]) * 100, 1)

    # 2. Schizophrenia vs. Bipolar Disorder (SCZ_VS_BD)
    # Check if raw authorized B-SNIP 1 data is available
    bsnip_available = False
    bsnip_count = 0
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM subjects WHERE dataset_id = 'bsnip' AND file_exists = 1 AND is_synthetic = 0")
            bsnip_count = cursor.fetchone()[0]
            bsnip_available = bsnip_count > 0
    except Exception:
        pass

    bsnip_dir = CHECKPOINTS_DIR.parent / "data" / "bsnip"
    raw_files = []
    if bsnip_dir.exists():
        raw_files = list(bsnip_dir.glob("**/*.edf")) + list(bsnip_dir.glob("**/*.bdf")) + list(bsnip_dir.glob("**/*.mat"))
    bsnip_data_present = bsnip_available or len(raw_files) > 0

    # Check if loadable SCZ vs BD checkpoint exists (never reuse ASEEG's best_model.pt or any ASEEG checkpoint)
    scz_bd_checkpoint_name = None
    scz_bd_arch = None
    scz_bd_status = "STATUS UNKNOWN"
    scz_bd_explanation = ""

    candidate_cp = None
    for cp in CHECKPOINTS_DIR.glob("*.pt"):
        if cp.name in ["best_model.pt", "aseeg_real_cnn_transformer.pt", "exp_real_4cbbe6c5_cnn_transformer.pt"]:
            continue
        try:
            st = torch.load(str(cp), map_location="cpu", weights_only=False)
            task = st.get("target_task", "")
            if "SCZ_VS_BD" in task and not st.get("is_synthetic", False) and st.get("dataset_id") == "bsnip":
                candidate_cp = (cp, st)
                break
        except Exception:
            pass

    if not bsnip_data_present:
        scz_bd_status = "DATA REQUIRED"
        scz_bd_explanation = "Authorized B-SNIP 1 data is required to train and evaluate this classification task."
    elif not candidate_cp:
        scz_bd_status = "MODEL NOT TRAINED"
        scz_bd_explanation = "Suitable authorized data is available, but a trained, loadable SCZ-vs-BD checkpoint is missing."
    else:
        # Check if real inference works
        cp_file, st = candidate_cp
        try:
            if "model_state_dict" in st:
                model = EEGCNNTransformer(num_channels=16, time_points=WINDOW_SAMPLES, num_classes=2)
                model.load_state_dict(st["model_state_dict"])
                model.eval()
                with torch.no_grad():
                    out = model(torch.zeros(1, 16, WINDOW_SAMPLES, dtype=torch.float32))
                if out.shape == (1, 2):
                    scz_bd_status = "MODEL READY"
                    scz_bd_explanation = "Authorized B-SNIP 1 data and trained SCZ-vs-BD checkpoint verified."
                    scz_bd_checkpoint_name = cp_file.name
                    scz_bd_arch = "CNN–Transformer"
                else:
                    scz_bd_status = "STATUS UNKNOWN"
                    scz_bd_explanation = f"SCZ-vs-BD model forward pass returned invalid output shape {out.shape}."
            else:
                scz_bd_status = "MODEL NOT TRAINED"
                scz_bd_explanation = "Candidate checkpoint is missing model_state_dict."
        except Exception as e:
            scz_bd_status = "STATUS UNKNOWN"
            scz_bd_explanation = f"Failed to verify SCZ-vs-BD model checkpoint: {str(e)}"

    # Check for saved evaluation results strictly belonging to SCZ vs BD (never reuse ASEEG)
    scz_bd_accuracy = None
    if scz_bd_status == "MODEL READY":
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT ev.accuracy
                FROM evaluations ev
                JOIN experiments e ON ev.experiment_id = e.id
                WHERE e.target_task = 'SCZ_VS_BD' AND e.dataset_id = 'bsnip' AND e.status = 'COMPLETED'
                ORDER BY ev.created_at DESC LIMIT 1
                """)
                row = cursor.fetchone()
                if row and row["accuracy"] is not None:
                    scz_bd_accuracy = round(float(row["accuracy"]) * 100, 1)
        except Exception:
            pass

    return {
        "sz_vs_hc": {
            "task_id": "SZ_VS_HC",
            "title": "Schizophrenia vs. Healthy Controls",
            "status": sz_hc_status,
            "status_reason": sz_hc_status_reason,
            "dataset": "ASEEG",
            "participants": aseeg_participants_count if aseeg_manifest_verified else None,
            "participants_verified": aseeg_manifest_verified,
            "sampling_rate": f"{int(aseeg_sampling_rate)} Hz" if (aseeg_manifest_verified and aseeg_sampling_rate) else None,
            "channels": aseeg_channels if aseeg_manifest_verified else None,
            "held_out_accuracy": f"{sz_hc_accuracy:.1f}%" if sz_hc_accuracy is not None else None,
            "active_checkpoint": sz_hc_active_checkpoint,
            "model_architecture": sz_hc_arch
        },
        "scz_vs_bd": {
            "task_id": "SCZ_VS_BD",
            "title": "Schizophrenia vs. Bipolar Disorder",
            "status": scz_bd_status,
            "status_reason": scz_bd_explanation,
            "explanation": scz_bd_explanation,
            "dataset": "B-SNIP 1",
            "participants": bsnip_count if bsnip_data_present else None,
            "participants_verified": False,
            "sampling_rate": None,
            "channels": None,
            "held_out_accuracy": f"{scz_bd_accuracy:.1f}%" if scz_bd_accuracy is not None else None,
            "active_checkpoint": scz_bd_checkpoint_name,
            "model_architecture": scz_bd_arch
        }
    }

@router.get("/classification-tasks")
def get_classification_tasks():
    """
    Direct endpoint for frontend to query live readiness and verified parameters of each classification task.
    """
    return compute_tasks_readiness()

@router.get("")
def get_system_health():
    """
    Returns system environment status, dependency versions, checkpoint status,
    supported modalities, dataset readiness summary, and classification tasks readiness.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints = [
        {
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "modified_at": f.stat().st_mtime
        }
        for f in CHECKPOINTS_DIR.glob("*") if f.is_file()
    ]
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, is_ready, verified_count, dataset_type FROM datasets")
        datasets = [dict(r) for r in cursor.fetchall()]

        cursor.execute("SELECT COUNT(*) FROM experiments WHERE status = 'COMPLETED'")
        completed_exps = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM preprocessing_jobs WHERE status = 'PROCESSED'")
        processed_jobs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM subjects WHERE dataset_id = 'demo' AND is_synthetic = 1")
        demo_count = cursor.fetchone()[0]

    tasks_readiness = compute_tasks_readiness()

    return {
        "status": "healthy",
        "system": {
            "torch_version": torch.__version__,
            "mne_version": str(getattr(mne, "__version__", "unknown")),
            "scipy_version": scipy.__version__,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "compute_device": device,
            "cuda_available": torch.cuda.is_available()
        },
        "models_status": {
            "eeg_encoder": "EEGCNNTransformer (16-ch, Conv1D + MultiHeadSelfAttention + AttentivePooler)",
            "mri_encoder": "StructuralMRIEncoder (14 Regional Volumetric & Cortical Thickness ROIs)",
            "clinical_encoder": "ClinicalEncoder (8 Neuropsychiatric Scales: PANSS, YMRS, MADRS, demographics)",
            "fusion_transformer": "MultimodalCrossAttentionFusion (Bidirectional Cross-Modal Self-Attention)",
            "baseline_model": "BaselineLogisticRegression (Spectral Features + L2 Regularization)",
            "checkpoints_available": len(checkpoints),
            "checkpoints": checkpoints
        },
        "classification_tasks_readiness": tasks_readiness,
        "datasets_status": datasets,
        "demo_cohort": {
            "ready": demo_count > 0,
            "synthetic_subjects_count": demo_count,
            "label": "[SYNTHETIC DEMO]"
        },
        "supported_input_formats": get_supported_formats_info()["supported_extensions"],
        "completed_experiments_count": completed_exps,
        "preprocessed_jobs_count": processed_jobs,
        "scientific_integrity_notice": (
            "Explainable Multimodal Transformer for SCZ-BD Differential Analysis. "
            "Engineering research prototype. Not certified for medical diagnosis or clinical decision support."
        )
    }
