from typing import Dict, Any, List, Optional
from pathlib import Path
from app.config import ASEEG_ZENODO_RECORD, ASEEG_ZENODO_URL, ASEEG_TITLE, DATA_DIR
from app.database import get_db

class ASEEGDatasetManager:
    """
    Manages the ASEEG Technical Validation Dataset (Zenodo 18029536).
    MANDATORY SCIENTIFIC CONSTRAINT:
    Strictly isolated from primary SCZ-vs-BD differential analysis.
    Task is explicitly designated and displayed as:
    'SZ vs HC Technical Validation'
    """
    DATASET_ID = "aseeg"

    @classmethod
    def get_provenance_info(cls) -> Dict[str, Any]:
        return {
            "zenodo_record": ASEEG_ZENODO_RECORD,
            "title": ASEEG_TITLE,
            "zenodo_url": ASEEG_ZENODO_URL,
            "task_label": "SZ vs HC Technical Validation",
            "license": "Creative Commons Attribution 4.0 International",
            "diagnostic_contrast": "Schizophrenia (SZ) vs Healthy Control (HC)",
            "purpose": (
                "Technical validation benchmark to verify signal processing, architecture convergence, "
                "and gradient-based explainability on open public EEG benchmarks. "
                "THIS MUST NEVER BE PRESENTED OR CONFLATED AS SCZ-VS-BD DIFFERENTIAL DIAGNOSIS PERFORMANCE."
            )
        }

    @classmethod
    def inspect_dataset_status(cls) -> Dict[str, Any]:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets WHERE id = 'aseeg'")
            row = cursor.fetchone()
            
            cursor.execute("""
            SELECT diagnostic_label, COUNT(*) as cnt
            FROM subjects
            WHERE dataset_id = 'aseeg' AND is_excluded = 0
            GROUP BY diagnostic_label
            """)
            dist_rows = cursor.fetchall()
            class_dist = {r["diagnostic_label"]: r["cnt"] for r in dist_rows}

        is_ready = bool(row["is_ready"]) if row else False
        return {
            "dataset_id": cls.DATASET_ID,
            "dataset_name": ASEEG_TITLE,
            "target_task": "SZ vs HC Technical Validation",
            "is_ready": is_ready,
            "verified_count": row["verified_count"] if row else 0,
            "class_distribution": class_dist,
            "provenance": cls.get_provenance_info(),
            "warning": "Technical validation benchmark only. Never interpret as BD-differential diagnosis."
        }
