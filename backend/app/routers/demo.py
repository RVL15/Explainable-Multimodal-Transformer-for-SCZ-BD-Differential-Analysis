from fastapi import APIRouter
from typing import Dict, Any
from app.eeg.synthetic import populate_demo_cohort
from app.database import get_db

router = APIRouter(prefix="/api/demo", tags=["Demo"])

@router.post("/populate")
def populate_synthetic_demo_data():
    """
    Populates synthetic demonstration cohort of SCZ, BD, and HC subjects.
    Every subject is explicitly marked with [SYNTHETIC DEMO] and is_synthetic = 1.
    """
    records = populate_demo_cohort(num_scz=18, num_bd=18, num_hc=12)
    return {
        "status": "success",
        "message": f"Successfully generated {len(records)} synthetic EEG recordings clearly labeled [SYNTHETIC DEMO].",
        "total_generated": len(records),
        "dataset_id": "demo",
        "notice": "SYNTHETIC DEMO DATA - Excluded from real scientific claims and medical conclusions."
    }

@router.get("/status")
def get_demo_status():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM subjects WHERE dataset_id = 'demo' AND is_synthetic = 1")
        demo_cnt = cursor.fetchone()[0]

    return {
        "demo_dataset_ready": demo_cnt > 0,
        "synthetic_subjects_count": demo_cnt,
        "marker": "[SYNTHETIC DEMO]",
        "purpose": "Faculty and technical demonstration when NIH NDA B-SNIP files are not present."
    }
