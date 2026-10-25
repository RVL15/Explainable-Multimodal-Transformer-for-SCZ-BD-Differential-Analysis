import uuid
import json
from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any
from app.database import get_db

router = APIRouter(prefix="/api/reports", tags=["Reports"])

@router.post("/generate")
def generate_research_report(experiment_id: Optional[str] = None):
    """
    Synthesizes a complete academic project report for Semester 7 faculty evaluation.
    Compiles actual dataset stats, preprocessing configuration, experimental results, and XAI findings.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Datasets status
        cursor.execute("SELECT * FROM datasets")
        datasets = [dict(r) for r in cursor.fetchall()]

        # Selected experiment or latest completed experiment
        if experiment_id:
            cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
            exp_row = cursor.fetchone()
        else:
            cursor.execute("SELECT * FROM experiments WHERE status = 'COMPLETED' ORDER BY created_at DESC LIMIT 1")
            exp_row = cursor.fetchone()

        # Evaluation results
        eval_data = None
        if exp_row:
            cursor.execute("SELECT * FROM evaluations WHERE experiment_id = ?", (exp_row["id"],))
            ev_row = cursor.fetchone()
            if ev_row:
                eval_data = dict(ev_row)
                eval_data["confusion_matrix"] = json.loads(eval_data["confusion_matrix"] or "[[0,0],[0,0]]")
                eval_data["subject_predictions"] = json.loads(eval_data["subject_predictions"] or "[]")

        # Baseline comparison
        baseline_data = None
        if exp_row and exp_row["dataset_id"]:
            cursor.execute("""
            SELECT ev.accuracy, ev.f1_score, ev.sensitivity, ev.specificity
            FROM experiments e
            JOIN evaluations ev ON e.id = ev.experiment_id
            WHERE e.dataset_id = ? AND e.model_type = 'LOGISTIC_REGRESSION' AND e.status = 'COMPLETED'
            ORDER BY e.created_at DESC LIMIT 1
            """, (exp_row["dataset_id"],))
            b_row = cursor.fetchone()
            if b_row:
                baseline_data = dict(b_row)

        report_id = f"rep_{uuid.uuid4().hex[:8]}"
        title = "Explainable Multimodal Transformer for SCZ-BD Differential Analysis: Final Research Report"
        
        report_payload = {
            "title": title,
            "project_name": "Explainable Multimodal Transformer for SCZ-BD Differential Analysis",
            "academic_context": "College Semester 7 Computer Engineering Research Project",
            "prototype_notice": "RESEARCH PROTOTYPE ONLY - NOT A MEDICAL DIAGNOSTIC SYSTEM",
            "generated_at": str(uuid.uuid4()),
            "datasets": datasets,
            "experiment": dict(exp_row) if exp_row else None,
            "evaluation": eval_data,
            "baseline": baseline_data,
            "methodology": {
                "eeg_pipeline": "1.0 - 45.0 Hz zero-phase Butterworth bandpass filter, Common Average Reference (CAR), channel z-score normalization, 2.0s windowing, artifact threshold rejection (150 uV).",
                "spectral_analysis": "Welch Power Spectral Density across Delta (0.5-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), Gamma (30-45Hz).",
                "deep_architecture": "Hybrid 1D Temporal/Spatial Convolutions + Multi-Head Self-Attention Transformer Encoders + Attentive Subject-Level Window Pooling.",
                "explainability": "Path-Integral Integrated Gradients attributing evidence directly to standard 10-20 channel electrodes and temporal windows."
            },
            "findings_summary": (
                f"Evaluation of {eval_data['test_subjects_count'] if eval_data else 0} held-out test subjects yielded "
                f"an accuracy of {eval_data['accuracy'] if eval_data else 'N/A'} (F1: {eval_data['f1_score'] if eval_data else 'N/A'}) "
                f"with an uncertainty rejection coverage of {eval_data['coverage'] if eval_data else 'N/A'} at threshold {eval_data['uncertainty_threshold'] if eval_data else 'N/A'}."
                if eval_data else "No completed experimental evaluation was attached."
            ),
            "scientific_limitations": [
                "Deep learning representations capture statistical correlations within recorded signals and do not represent validated medical etiologies.",
                "Cross-site phenotypic variability and medication effects (antipsychotics vs mood stabilizers) can confound spectral signatures.",
                "Strict subject-level partitioning was enforced to prevent window leakage; results must be replicated on larger independent cohorts.",
                "Uncertainty thresholding correctly rejected low-confidence cases to preserve diagnostic safety margin."
            ]
        }

        cursor.execute("""
        INSERT INTO research_reports (id, title, experiment_id, summary, report_data)
        VALUES (?, ?, ?, ?, ?)
        """, (
            report_id,
            title,
            exp_row["id"] if exp_row else None,
            report_payload["findings_summary"],
            json.dumps(report_payload)
        ))

        return {
            "id": report_id,
            "title": title,
            "summary": report_payload["findings_summary"],
            "report_data": report_payload
        }

@router.get("")
def list_reports():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, experiment_id, summary, created_at FROM research_reports ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

@router.get("/{report_id}")
def get_report(report_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM research_reports WHERE id = ?", (report_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found.")

        d = dict(row)
        d["report_data"] = json.loads(d["report_data"] or "{}")
        return d
