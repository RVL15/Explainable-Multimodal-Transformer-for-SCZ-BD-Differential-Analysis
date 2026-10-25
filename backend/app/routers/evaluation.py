import json
import io
import csv
from fastapi import APIRouter, HTTPException, Response
from typing import Optional, List, Dict, Any
from app.database import get_db

router = APIRouter(prefix="/api/evaluation", tags=["Evaluation"])

@router.get("/{experiment_id}")
def get_experiment_evaluation(experiment_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM evaluations WHERE experiment_id = ?", (experiment_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"No evaluation results recorded for experiment {experiment_id}.")

        ev = dict(row)
        ev["confusion_matrix"] = json.loads(ev["confusion_matrix"] or "[[0,0],[0,0]]")
        ev["subject_predictions"] = json.loads(ev["subject_predictions"] or "[]")

        # Fetch experiment details for context and baseline comparison
        cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        exp_row = cursor.fetchone()
        if exp_row:
            ev["model_type"] = exp_row["model_type"]
            ev["train_loss_history"] = json.loads(exp_row["train_loss_history"] or "[]")
            ev["val_loss_history"] = json.loads(exp_row["val_loss_history"] or "[]")

        # Baseline comparison: find most recent baseline experiment on same dataset
        cursor.execute("""
        SELECT e.*, ev.accuracy, ev.f1_score, ev.sensitivity, ev.specificity
        FROM experiments e
        JOIN evaluations ev ON e.id = ev.experiment_id
        WHERE e.dataset_id = ? AND e.model_type = 'LOGISTIC_REGRESSION' AND e.status = 'COMPLETED'
        ORDER BY e.created_at DESC LIMIT 1
        """, (ev["dataset_id"],))
        baseline_row = cursor.fetchone()
        
        if baseline_row and ev["model_type"] != "LOGISTIC_REGRESSION":
            ev["baseline_comparison"] = {
                "baseline_model": "Logistic Regression (Handcrafted Spectral Features)",
                "baseline_accuracy": baseline_row["accuracy"],
                "baseline_f1": baseline_row["f1_score"],
                "baseline_sensitivity": baseline_row["sensitivity"],
                "baseline_specificity": baseline_row["specificity"],
                "delta_accuracy": round(ev["accuracy"] - baseline_row["accuracy"], 4) if ev["accuracy"] and baseline_row["accuracy"] else 0.0
            }
        else:
            ev["baseline_comparison"] = None

        return ev

@router.get("/latest/summary")
def get_latest_evaluation():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT ev.*, e.name as experiment_name, e.model_type
        FROM evaluations ev
        JOIN experiments e ON ev.experiment_id = e.id
        ORDER BY ev.created_at DESC LIMIT 1
        """)
        row = cursor.fetchone()
        if not row:
            return {"has_evaluation": False, "message": "No evaluations completed yet."}

        ev = dict(row)
        ev["has_evaluation"] = True
        ev["confusion_matrix"] = json.loads(ev["confusion_matrix"] or "[[0,0],[0,0]]")
        ev["subject_predictions"] = json.loads(ev["subject_predictions"] or "[]")
        return ev

@router.get("/{experiment_id}/export/csv")
def export_predictions_csv(experiment_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM evaluations WHERE experiment_id = ?", (experiment_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Evaluation not found.")

        predictions = json.loads(row["subject_predictions"] or "[]")
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["subject_id", "true_label", "predicted_label", "prob_scz", "prob_bd", "is_uncertain", "is_correct"])

        for p in predictions:
            writer.writerow([
                p.get("subject_id"),
                p.get("true_label"),
                p.get("predicted_label"),
                p.get("prob_scz"),
                p.get("prob_bd"),
                p.get("is_uncertain"),
                p.get("is_correct")
            ])

        csv_data = output.getvalue()
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=evaluation_{experiment_id}_predictions.csv"}
        )

@router.get("/{experiment_id}/export/json")
def export_evaluation_json(experiment_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM evaluations WHERE experiment_id = ?", (experiment_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Evaluation not found.")

        ev = dict(row)
        ev["confusion_matrix"] = json.loads(ev["confusion_matrix"] or "[[0,0],[0,0]]")
        ev["subject_predictions"] = json.loads(ev["subject_predictions"] or "[]")

        json_str = json.dumps(ev, indent=2)
        return Response(
            content=json_str,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=evaluation_{experiment_id}.json"}
        )
