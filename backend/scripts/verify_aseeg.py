import sys
sys.path.append("backend")
import os
import glob
import numpy as np
from app.eeg.io import load_eeg_file, get_eeg_file_metadata
from app.database import get_db

with get_db() as conn:
    c = conn.cursor()
    c.execute("SELECT subject_id, diagnostic_label, recording_path, file_exists FROM subjects WHERE dataset_id='aseeg'")
    rows = [dict(r) for r in c.fetchall()]

print(f"Total ASEEG subjects in DB: {len(rows)}")
missing = 0
for r in rows:
    if not os.path.exists(r["recording_path"]):
        missing += 1
print(f"Missing recordings on disk: {missing}")

# Check first 3 files
for r in rows[:3]:
    p = r["recording_path"]
    meta = get_eeg_file_metadata(p)
    print(f"{r['subject_id']} ({r['diagnostic_label']}): {meta['channels_count']} ch, {meta['sample_rate']} Hz, {meta['duration_sec']}s, size={meta['file_size_bytes']} bytes")
