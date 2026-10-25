import sqlite3
import json

conn = sqlite3.connect("backend/research.db")
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("=== ALL EVALUATIONS ===")
for row in cursor.execute("SELECT * FROM evaluations").fetchall():
    d = dict(row)
    preds = json.loads(d["subject_predictions"]) if d.get("subject_predictions") else []
    print(f"Eval ID: {d['id']}, Exp: {d['experiment_id']}, Task: {d['target_task']}")
    print(f"  Test count: {d['test_subjects_count']}, Acc: {d['accuracy']}, Prec: {d.get('precision')}, Rec: {d.get('recall')}, F1: {d['f1_score']}, Sens: {d.get('sensitivity')}, Spec: {d.get('specificity')}, ROC_AUC: {d['roc_auc']}")
    print(f"  Coverage: {d['coverage']}, Uncertain count: {d['uncertain_count']}, Matrix: {d['confusion_matrix']}")
    print(f"  Saved subject predictions count: {len(preds)}")
    if preds:
        print(f"  First pred: {preds[0]}")
