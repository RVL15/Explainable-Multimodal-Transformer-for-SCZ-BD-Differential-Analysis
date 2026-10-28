import csv
import io
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from app.config import BSNIP_NDA_COLLECTION_ID, BSNIP_NDA_URL, BSNIP_TITLE, DATA_DIR
from app.database import get_db

class BSNIPDatasetManager:
    """
    Manages B-SNIP 1 dataset provenance, manifest verification, and recording validation.
    Strictly adheres to NIH NDA data use compliance.
    """
    DATASET_ID = "bsnip1"

    @classmethod
    def get_provenance_info(cls) -> Dict[str, Any]:
        return {
            "collection_id": BSNIP_NDA_COLLECTION_ID,
            "title": BSNIP_TITLE,
            "nda_url": BSNIP_NDA_URL,
            "license": "NIH NDA Data Use Certification (DUC) Required",
            "study_design": "Multi-site cross-diagnostic phenotypic study (Schizophrenia, Schizoaffective, Bipolar Disorder I, and Controls)",
            "acquisition_sites": ["Baltimore (UMD)", "Boston (BIDMC/Harvard)", "Chicago (UIC)", "Dallas (UTSW)", "Hartford (IOL)"],
            "recording_modality": "Resting-state & auditory evoked potential EEG (64/128-channel sensor nets downsampled to standard 10-20 montage)",
            "compliance_notice": (
                "B-SNIP 1 data is protected under NIH NDA privacy agreements. "
                "Researchers must obtain approval via nda.nih.gov. The application verifies "
                "locally authorized manifests and raw EDF/FIF files without bypassing access restrictions."
            )
        }

    @classmethod
    def validate_manifest_content(cls, csv_text: str, data_directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Parses and validates a user-provided CSV manifest for B-SNIP subjects.
        Detects missing files, duplicate subjects, invalid labels, and builds exclusion reports.
        """
        f = io.StringIO(csv_text)
        reader = csv.DictReader(f)
        
        # Standardize field names
        headers = [h.strip().lower() for h in (reader.fieldnames or [])]
        
        id_col = next((c for c in headers if c in ["subject_id", "src_subject_id", "subid", "id", "subject"]), None)
        label_col = next((c for c in headers if c in ["diagnosis", "phenotype", "dx", "group", "label"]), None)
        file_col = next((c for c in headers if c in ["recording_file", "file_name", "filename", "filepath", "path", "eeg_file"]), None)
        age_col = next((c for c in headers if c in ["age", "interview_age", "age_years"]), None)
        sex_col = next((c for c in headers if c in ["sex", "gender"]), None)
        site_col = next((c for c in headers if c in ["site", "study_site", "institution"]), None)

        if not id_col or not label_col:
            raise ValueError(f"Manifest missing required columns. Must include subject identifier and diagnosis columns. Found: {reader.fieldnames}")

        seen_subjects = set()
        duplicates = []
        exclusions = []
        valid_records = []
        missing_files = []
        class_dist = {"SCZ": 0, "BD": 0, "HC": 0, "OTHER": 0}

        # Root search dir
        search_dir = Path(data_directory) if data_directory else (DATA_DIR / "bsnip")

        f.seek(0)
        # re-create dictreader with original fieldnames
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader, start=1):
            # Normalize row keys to lowercase
            norm_row = {k.strip().lower(): v.strip() for k, v in row.items() if k}
            sub_id = norm_row.get(id_col, "").strip()
            raw_label = norm_row.get(label_col, "").strip().upper()
            
            if not sub_id:
                exclusions.append({
                    "row": row_idx,
                    "subject_id": "UNKNOWN",
                    "reason": "Missing subject ID value."
                })
                continue

            # Duplicate check
            if sub_id in seen_subjects:
                duplicates.append(sub_id)
                exclusions.append({
                    "row": row_idx,
                    "subject_id": sub_id,
                    "reason": "Duplicate subject ID in manifest."
                })
                continue
            seen_subjects.add(sub_id)

            # Map label
            if raw_label in ["SCZ", "SCHIZOPHRENIA", "SZ", "1"]:
                diag = "SCZ"
            elif raw_label in ["BD", "BIPOLAR", "BP", "BIPOLAR_I", "2"]:
                diag = "BD"
            elif raw_label in ["HC", "HEALTHY", "CONTROL", "NC", "0"]:
                diag = "HC"
            else:
                diag = "OTHER"
                exclusions.append({
                    "row": row_idx,
                    "subject_id": sub_id,
                    "reason": f"Unrecognized diagnosis label '{raw_label}'. Differential analysis requires SCZ or BD."
                })
                continue

            class_dist[diag] = class_dist.get(diag, 0) + 1

            # Check file existence
            rec_filename = norm_row.get(file_col, f"{sub_id}.edf") if file_col else f"{sub_id}.edf"
            candidate_path = Path(rec_filename)
            if not candidate_path.is_absolute():
                candidate_path = search_dir / rec_filename

            file_exists = candidate_path.exists()
            if not file_exists:
                # Also try checking with other common extensions
                for alt_ext in [".edf", ".fif", ".mat", ".npy"]:
                    alt_path = search_dir / f"{sub_id}{alt_ext}"
                    if alt_path.exists():
                        candidate_path = alt_path
                        file_exists = True
                        break

            if not file_exists:
                missing_files.append(sub_id)
                exclusions.append({
                    "row": row_idx,
                    "subject_id": sub_id,
                    "reason": f"Raw EEG recording file not located at {candidate_path}."
                })
                continue

            # Age and sex
            age_val = None
            if age_col:
                raw_age = norm_row.get(age_col)
                if raw_age is not None and str(raw_age).strip():
                    try:
                        age_val = float(str(raw_age).strip())
                    except ValueError:
                        age_val = None

            sex_val = norm_row.get(sex_col, "U").upper() if sex_col else "U"
            site_val = norm_row.get(site_col, "Unknown") if site_col else "Unknown"

            valid_records.append({
                "subject_id": sub_id,
                "dataset_id": cls.DATASET_ID,
                "diagnostic_label": diag,
                "age": age_val,
                "sex": sex_val,
                "site": site_val,
                "recording_path": str(candidate_path),
                "file_exists": 1,
                "is_excluded": 0,
                "is_synthetic": 0
            })

        total_manifest_records = row_idx
        is_ready = (len(valid_records) >= 10 and class_dist["SCZ"] >= 5 and class_dist["BD"] >= 5)
        
        status_msg = (
            f"Validated {len(valid_records)} verified subjects ({class_dist['SCZ']} SCZ, {class_dist['BD']} BD). Dataset ready for training."
            if is_ready else
            "B-SNIP 1 primary cohort is currently UNAVAILABLE. Valid EDF/FIF recordings must be placed in data directory or configured via NIH NDA collection #2274."
        )

        # Store in database
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Update datasets table
            cursor.execute("""
            UPDATE datasets SET
                is_ready = ?,
                verified_count = ?,
                missing_count = ?,
                duplicate_count = ?,
                excluded_count = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 'bsnip1'
            """, (
                1 if is_ready else 0,
                len(valid_records),
                len(missing_files),
                len(duplicates),
                len(exclusions)
            ))

            # Insert/update valid subjects
            for rec in valid_records:
                cursor.execute("""
                INSERT INTO subjects (
                    subject_id, dataset_id, diagnostic_label, age, sex, site,
                    recording_path, file_exists, is_excluded, is_synthetic
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
                ON CONFLICT(dataset_id, subject_id) DO UPDATE SET
                    diagnostic_label = excluded.diagnostic_label,
                    age = excluded.age,
                    sex = excluded.sex,
                    site = excluded.site,
                    recording_path = excluded.recording_path,
                    file_exists = excluded.file_exists,
                    is_excluded = 0
                """, (
                    rec["subject_id"], rec["dataset_id"], rec["diagnostic_label"],
                    rec["age"], rec["sex"], rec["site"], rec["recording_path"], rec["file_exists"]
                ))

        return {
            "dataset_id": cls.DATASET_ID,
            "total_rows": total_manifest_records,
            "total_manifest_records": total_manifest_records,
            "valid_subjects_count": len(valid_records),
            "missing_files_count": len(missing_files),
            "duplicate_subjects_count": len(duplicates),
            "duplicates": duplicates,
            "excluded_subjects_count": len(exclusions),
            "class_distribution": class_dist,
            "exclusions": exclusions[:50],  # Return up to 50 for inspection
            "readiness_status": "READY" if is_ready else "UNAVAILABLE_MISSING_DATA",
            "message": status_msg
        }
