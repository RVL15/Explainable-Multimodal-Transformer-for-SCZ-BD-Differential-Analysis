from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, List, Dict, Any
from app.database import get_db
from app.datasets.bsnip_manager import BSNIPDatasetManager
from app.datasets.aseeg_manager import ASEEGDatasetManager

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])

@router.get("")
def list_datasets():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM datasets ORDER BY id ASC")
        rows = cursor.fetchall()
        
        result = []
        for r in rows:
            d = dict(r)
            d["is_ready"] = bool(d["is_ready"])
            result.append(d)
        return result

@router.get("/{dataset_id}")
def get_dataset_details(dataset_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found.")

        d = dict(row)
        d["is_ready"] = bool(d["is_ready"])

        if dataset_id == "bsnip1":
            d["provenance"] = BSNIPDatasetManager.get_provenance_info()
        elif dataset_id == "aseeg":
            d["provenance"] = ASEEGDatasetManager.get_provenance_info()

        # Class breakdown
        cursor.execute("""
        SELECT diagnostic_label, COUNT(*) as cnt
        FROM subjects
        WHERE dataset_id = ? AND is_excluded = 0
        GROUP BY diagnostic_label
        """, (dataset_id,))
        d["class_counts"] = {r["diagnostic_label"]: r["cnt"] for r in cursor.fetchall()}

        return d

@router.post("/bsnip/validate-manifest")
async def validate_bsnip_manifest(
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    data_directory: Optional[str] = Form(None)
):
    """
    Validates user-provided CSV manifest for the B-SNIP 1 dataset.
    Detects missing recordings on disk, duplicate subject entries, and invalid diagnostic labels.
    """
    content = ""
    if file:
        content_bytes = await file.read()
        content = content_bytes.decode("utf-8", errors="replace")
    elif csv_text:
        content = csv_text
    else:
        raise HTTPException(status_code=400, detail="Either CSV file or csv_text must be provided.")

    try:
        res = BSNIPDatasetManager.validate_manifest_content(content, data_directory=data_directory)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{dataset_id}/subjects")
def get_dataset_subjects(
    dataset_id: str,
    diagnostic_label: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    with get_db() as conn:
        cursor = conn.cursor()
        
        query = "SELECT * FROM subjects WHERE dataset_id = ?"
        params: List[Any] = [dataset_id]

        if diagnostic_label:
            query += " AND diagnostic_label = ?"
            params.append(diagnostic_label)

        query += " ORDER BY id ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        subjects = []
        for r in rows:
            sub = dict(r)
            sub["file_exists"] = bool(sub["file_exists"])
            sub["is_excluded"] = bool(sub["is_excluded"])
            sub["is_synthetic"] = bool(sub["is_synthetic"])
            subjects.append(sub)

        cursor.execute("SELECT COUNT(*) FROM subjects WHERE dataset_id = ?", (dataset_id,))
        total_count = cursor.fetchone()[0]

        return {
            "dataset_id": dataset_id,
            "total_count": total_count,
            "subjects": subjects
        }

from pydantic import BaseModel

class DirectoryIngestionRequest(BaseModel):
    directory_path: str
    dataset_id: str = "custom"
    dataset_name: Optional[str] = None
    target_task: Optional[str] = None

@router.post("/ingest-directory")
def ingest_directory(req: DirectoryIngestionRequest):
    """
    Scans local directory containing downloaded, authorized EEG dataset files.
    Discovers recordings, maps participant metadata, validates signals, and records exclusions.
    """
    from app.datasets.ingestion import DatasetIngestionEngine
    try:
        summary = DatasetIngestionEngine.scan_and_ingest_directory(
            directory_path=req.directory_path,
            dataset_id=req.dataset_id,
            dataset_name=req.dataset_name,
            target_task=req.target_task
        )
        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{dataset_id}/ingestion-progress")
def get_dataset_ingestion_progress(dataset_id: str):
    """
    Returns real-time streaming ingestion progress, processed file count, exclusions, and completion status.
    """
    from app.datasets.ingestion import DatasetIngestionEngine
    return DatasetIngestionEngine.get_ingestion_progress(dataset_id)

@router.get("/{dataset_id}/manifest")
def get_dataset_manifest(dataset_id: str):
    """
    Returns dataset manifest metadata, total records, and sample entries.
    """
    import csv
    from app.datasets.ingestion import DatasetIngestionEngine
    manifest_path = DatasetIngestionEngine.get_manifest_path(dataset_id)
    if not manifest_path or not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"No manifest generated for dataset '{dataset_id}'. Run ingestion first.")

    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    return {
        "dataset_id": dataset_id,
        "manifest_path": str(manifest_path),
        "total_records": len(rows),
        "validated_count": len([r for r in rows if r.get("status") == "VALIDATED"]),
        "excluded_count": len([r for r in rows if r.get("status") in ["EXCLUDED", "ERROR", "DUPLICATE"]]),
        "preview_rows": rows[:15],
        "download_url": f"/api/datasets/{dataset_id}/manifest/download"
    }

@router.get("/{dataset_id}/manifest/download")
def download_dataset_manifest(dataset_id: str):
    """
    Downloads the genuine dataset manifest CSV file.
    """
    from fastapi.responses import FileResponse
    from app.datasets.ingestion import DatasetIngestionEngine
    manifest_path = DatasetIngestionEngine.get_manifest_path(dataset_id)
    if not manifest_path or not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"No manifest found for dataset '{dataset_id}'.")

    return FileResponse(
        path=str(manifest_path),
        media_type="text/csv",
        filename=f"{dataset_id}_manifest.csv"
    )

