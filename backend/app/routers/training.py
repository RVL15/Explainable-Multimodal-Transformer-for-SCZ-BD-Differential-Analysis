import uuid
import json
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.database import get_db
from app.schemas import TrainModelRequest
from app.models.pipeline import start_training_in_background

router = APIRouter(prefix="/api/training", tags=["Training"])

@router.post("/start")
def start_experiment(req: TrainModelRequest):
    """
    Launches a background training experiment for CNN_TRANSFORMER or LOGISTIC_REGRESSION.
    Enforces strict subject-level isolation to prevent data leakage.
    """
    # Verify dataset readiness
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM datasets WHERE id = ?", (req.dataset_id,))
        dataset_row = cursor.fetchone()
        if not dataset_row:
            raise HTTPException(status_code=404, detail=f"Dataset {req.dataset_id} not found.")

        # Check subject counts for target task
        cursor.execute("""
        SELECT diagnostic_label, COUNT(*) as cnt
        FROM subjects
        WHERE dataset_id = ? AND is_excluded = 0 AND file_exists = 1
        GROUP BY diagnostic_label
        """, (req.dataset_id,))
        counts = {r["diagnostic_label"]: r["cnt"] for r in cursor.fetchall()}

        if req.target_task == "SCZ_VS_BD":
            if counts.get("SCZ", 0) < 3 or counts.get("BD", 0) < 3:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Cannot start SCZ-vs-BD training on '{req.dataset_id}'. "
                        f"Found {counts.get('SCZ', 0)} SCZ and {counts.get('BD', 0)} BD subjects. "
                        "At least 3 verified subjects per class are required. "
                        "Configure valid B-SNIP data or use the Synthetic Demo cohort."
                    )
                )
        else: # SZ vs HC Technical Validation
            if counts.get("SZ", 0) + counts.get("SCZ", 0) < 3 or counts.get("HC", 0) < 3:
                raise HTTPException(
                    status_code=400,
                    detail="Insufficient subjects for SZ vs HC Technical Validation benchmark."
                )

        experiment_id = f"exp_{req.model_type.lower()}_{uuid.uuid4().hex[:8]}"
        exp_name = f"{req.model_type} on {req.dataset_id.upper()} ({req.target_task})"

        hyperparams = req.dict()

        cursor.execute("""
        INSERT INTO experiments (
            id, name, model_type, dataset_id, target_task, status,
            current_epoch, total_epochs, hyperparameters,
            train_loss_history, val_loss_history, val_metric_history
        ) VALUES (?, ?, ?, ?, ?, 'RUNNING', 0, ?, ?, '[]', '[]', '[]')
        """, (
            experiment_id,
            exp_name,
            req.model_type,
            req.dataset_id,
            req.target_task,
            req.epochs if req.model_type == "CNN_TRANSFORMER" else 3,
            json.dumps(hyperparams)
        ))

    # Launch background job
    start_training_in_background(
        experiment_id=experiment_id,
        dataset_id=req.dataset_id,
        target_task=req.target_task,
        model_type=req.model_type,
        hyperparams=hyperparams
    )

    return {
        "experiment_id": experiment_id,
        "name": exp_name,
        "status": "RUNNING",
        "message": f"Training job launched in background for {req.model_type}."
    }

@router.get("/experiments")
def list_experiments():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments ORDER BY created_at DESC")
        rows = cursor.fetchall()
        
        experiments = []
        for r in rows:
            exp = dict(r)
            exp["train_loss_history"] = json.loads(exp["train_loss_history"] or "[]")
            exp["val_loss_history"] = json.loads(exp["val_loss_history"] or "[]")
            exp["val_metric_history"] = json.loads(exp["val_metric_history"] or "[]")
            exp["final_metrics"] = json.loads(exp["final_metrics"]) if exp["final_metrics"] else None
            exp["hyperparameters"] = json.loads(exp["hyperparameters"]) if exp["hyperparameters"] else None
            experiments.append(exp)

        return experiments

@router.get("/experiments/{experiment_id}")
def get_experiment_details(experiment_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found.")

        exp = dict(row)
        exp["train_loss_history"] = json.loads(exp["train_loss_history"] or "[]")
        exp["val_loss_history"] = json.loads(exp["val_loss_history"] or "[]")
        exp["val_metric_history"] = json.loads(exp["val_metric_history"] or "[]")
        exp["final_metrics"] = json.loads(exp["final_metrics"]) if exp["final_metrics"] else None
        exp["hyperparameters"] = json.loads(exp["hyperparameters"]) if exp["hyperparameters"] else None

        # Check if evaluation exists
        cursor.execute("SELECT id FROM evaluations WHERE experiment_id = ?", (experiment_id,))
        eval_row = cursor.fetchone()
        exp["evaluation_id"] = eval_row["id"] if eval_row else None

        return exp
