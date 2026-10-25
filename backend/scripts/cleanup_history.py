import sys
sys.path.append("backend")
from app.database import get_db

with get_db() as conn:
    c = conn.cursor()
    c.execute("DELETE FROM analysis_history WHERE target_task='SCZ_VS_BD' AND (filename LIKE '%UPLOAD_n%' OR filename LIKE '%UPLOAD_sz%')")
    deleted = c.rowcount
    conn.commit()
    print("Cleaned up HC/SZ in SCZ_VS_BD:", deleted)
