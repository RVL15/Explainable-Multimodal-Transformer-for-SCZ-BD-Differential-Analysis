import sqlite3

conn = sqlite3.connect("backend/research.db")
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

for t in ["datasets", "experiments", "subjects", "analysis_history"]:
    print(f"--- Columns in {t} ---")
    cols = [r[1] for r in cursor.execute(f"PRAGMA table_info({t})").fetchall()]
    print(cols)

print("--- Datasets ---")
for r in cursor.execute("SELECT * FROM datasets").fetchall():
    print(dict(r))

print("--- Experiments ---")
for r in cursor.execute("SELECT id, name, dataset_id, target_task, status FROM experiments").fetchall():
    print(dict(r))

print("--- Subjects summary by dataset and diagnosis ---")
for r in cursor.execute("SELECT dataset_id, diagnostic_label, count(*), is_synthetic FROM subjects GROUP BY dataset_id, diagnostic_label, is_synthetic").fetchall():
    print(dict(r))
