import io
import csv
import json
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.database import get_db

router = APIRouter(prefix="/api/history", tags=["History"])

@router.get("")
def list_analysis_history(
    search: Optional[str] = Query(None, description="Search by filename or diagnosis"),
    target_task: Optional[str] = Query(None, description="Filter by classification task"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    """
    Returns searchable history of completed EEG and multimodal predictions.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM analysis_history WHERE 1=1"
        params: List[Any] = []

        if target_task:
            if target_task in ["SZ_VS_HC", "SZ_VS_HC_TECH_VAL"]:
                query += " AND target_task IN ('SZ_VS_HC', 'SZ_VS_HC_TECH_VAL')"
            else:
                query += " AND target_task = ?"
                params.append(target_task)

        if search:
            query += " AND (filename LIKE ? OR predicted_label LIKE ? OR id LIKE ?)"
            term = f"%{search}%"
            params.extend([term, term, term])

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        records = []
        for r in rows:
            rec = dict(r)
            rec["is_uncertain"] = bool(rec["is_uncertain"])
            # Provide task-correct probabilities
            p_0 = rec.get("prob_scz", 0.0)
            p_1 = rec.get("prob_bd", 0.0)
            rec["prob_sz"] = p_0
            rec["prob_hc"] = p_1
            try:
                rec["top_channels"] = json.loads(rec["top_channels"] or "[]")
            except Exception:
                rec["top_channels"] = []
            records.append(rec)

        cursor.execute("SELECT COUNT(*) FROM analysis_history")
        total_count = cursor.fetchone()[0]

        return {
            "total_count": total_count,
            "records": records
        }

@router.get("/export/csv")
def export_history_csv():
    """
    Exports all analysis history records to CSV format.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analysis_history ORDER BY created_at DESC")
        rows = cursor.fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "filename", "dataset_id", "target_task", "model_version",
        "predicted_label", "raw_label", "prob_scz", "prob_bd", "confidence",
        "is_uncertain", "evidence_margin", "created_at"
    ])

    for r in rows:
        writer.writerow([
            r["id"], r["filename"], r["dataset_id"], r["target_task"], r["model_version"],
            r["predicted_label"], r["raw_label"], r["prob_scz"], r["prob_bd"], r["confidence"],
            r["is_uncertain"], r["evidence_margin"], r["created_at"]
        ])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=eeg_analysis_history.csv"}
    )

@router.get("/{record_id}")
def get_history_record(record_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analysis_history WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Analysis record not found.")

        rec = dict(row)
        rec["is_uncertain"] = bool(rec["is_uncertain"])
        rec["prob_sz"] = rec.get("prob_scz", 0.0)
        rec["prob_hc"] = rec.get("prob_bd", 0.0)
        try:
            rec["top_channels"] = json.loads(rec["top_channels"] or "[]")
        except Exception:
            rec["top_channels"] = []
        return rec

@router.delete("/{record_id}")
def delete_history_record(record_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Analysis record not found.")
        return {"status": "deleted", "id": record_id}

# Recent Activity Router
activity_router = APIRouter(prefix="/api/activity", tags=["Activity"])

@activity_router.get("/recent")
def get_recent_activity(limit: int = 10):
    """
    Aggregates genuine chronological system events from experiments, evaluations,
    dataset scans, and predictions.
    """
    events: List[Dict[str, Any]] = []

    with get_db() as conn:
        cursor = conn.cursor()

        # 1. Experiments
        cursor.execute("""
        SELECT id, name, model_type, target_task, status, created_at, finished_at
        FROM experiments
        ORDER BY created_at DESC LIMIT 5
        """)
        for r in cursor.fetchall():
            events.append({
                "id": f"exp_{r['id']}",
                "type": "TRAINING",
                "title": f"Model Training: {r['name']}",
                "description": f"Architecture: {r['model_type']} • Task: {r['target_task']} • Status: {r['status']}",
                "timestamp": r["finished_at"] or r["created_at"],
                "status": r["status"]
            })

        # 2. Evaluations
        cursor.execute("""
        SELECT id, experiment_id, dataset_id, target_task, accuracy, f1_score, test_subjects_count, created_at
        FROM evaluations
        ORDER BY created_at DESC LIMIT 5
        """)
        for r in cursor.fetchall():
            acc_pct = f"{round(r['accuracy'] * 100, 1)}%" if r["accuracy"] is not None else "N/A"
            events.append({
                "id": f"eval_{r['id']}",
                "type": "EVALUATION",
                "title": f"Held-Out Evaluation: {acc_pct} Accuracy",
                "description": f"Evaluated on {r['test_subjects_count']} unseen test subjects in {r['dataset_id'].upper()}",
                "timestamp": r["created_at"],
                "status": "COMPLETED"
            })

        # 3. Live EEG Analysis History
        cursor.execute("""
        SELECT id, filename, target_task, predicted_label, confidence, created_at
        FROM analysis_history
        ORDER BY created_at DESC LIMIT 5
        """)
        for r in cursor.fetchall():
            conf_pct = f"{round(r['confidence'] * 100, 1)}%" if r["confidence"] is not None else ""
            events.append({
                "id": f"pred_{r['id']}",
                "type": "ANALYSIS",
                "title": f"EEG Analysis: {r['predicted_label']} ({conf_pct})",
                "description": f"File: {r['filename']} • Task: {r['target_task']}",
                "timestamp": r["created_at"],
                "status": "COMPLETED"
            })

        # 4. Datasets
        cursor.execute("""
        SELECT id, name, verified_count, updated_at
        FROM datasets
        WHERE verified_count > 0
        ORDER BY updated_at DESC LIMIT 3
        """)
        for r in cursor.fetchall():
            events.append({
                "id": f"ds_{r['id']}",
                "type": "DATASET",
                "title": f"Dataset Synchronized: {r['name']}",
                "description": f"{r['verified_count']} verified multi-channel recordings registered",
                "timestamp": r["updated_at"],
                "status": "READY"
            })

    # Sort all events chronologically descending
    events.sort(key=lambda x: str(x.get("timestamp") or ""), reverse=True)
    return events[:limit]
