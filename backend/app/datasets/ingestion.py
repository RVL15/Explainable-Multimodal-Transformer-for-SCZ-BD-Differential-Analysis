import os
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import pandas as pd

from app.database import get_db
from app.config import DEFAULT_SAMPLE_RATE
from app.eeg.io import load_eeg_file
from app.eeg.validation import validate_eeg_array, SUPPORTED_EXTENSIONS

# Global active ingestion jobs for streaming progress telemetry
active_ingestion_jobs: Dict[str, Dict[str, Any]] = {}

class DatasetIngestionEngine:
    """
    Automated dataset ingestion, discovery, metadata mapping, and validation engine.
    Supports chunked, streaming ingestion for multi-gigabyte EEG datasets (e.g., 26 GB),
    resumable processing across interruptions, manifest generation, and live progress telemetry.
    """

    @classmethod
    def get_ingestion_progress(cls, dataset_id: str) -> Dict[str, Any]:
        """Returns the current streaming ingestion telemetry for the given dataset."""
        if dataset_id in active_ingestion_jobs:
            return active_ingestion_jobs[dataset_id]
        # Fallback check from database
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                return {
                    "dataset_id": dataset_id,
                    "status": "COMPLETED" if d["is_ready"] else "IDLE",
                    "total_files": d["verified_count"] + d["excluded_count"],
                    "processed_files": d["verified_count"] + d["excluded_count"],
                    "valid_count": d["verified_count"],
                    "exclusions_count": d["excluded_count"],
                    "progress_percent": 100.0 if d["is_ready"] else 0.0,
                    "current_file": None,
                    "is_ready_for_training": bool(d["is_ready"])
                }
        return {
            "dataset_id": dataset_id,
            "status": "IDLE",
            "total_files": 0,
            "processed_files": 0,
            "valid_count": 0,
            "exclusions_count": 0,
            "progress_percent": 0.0,
            "current_file": None,
            "is_ready_for_training": False
        }

    @classmethod
    def get_manifest_path(cls, dataset_id: str) -> Optional[Path]:
        """Returns the filesystem path to the generated dataset manifest CSV."""
        from app.config import MANIFESTS_DIR
        path = MANIFESTS_DIR / f"{dataset_id}_manifest.csv"
        return path if path.exists() else None

    @classmethod
    def scan_and_ingest_directory(
        cls,
        directory_path: str,
        dataset_id: str = "custom",
        dataset_name: Optional[str] = None,
        target_task: Optional[str] = None,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Scans directory containing downloaded, authorized dataset files.
        Processes in manageable chunks to avoid loading multi-gigabyte datasets into RAM.
        Resumable after interruption: previously validated files in the manifest are preserved.
        Generates and flushes a complete manifest CSV at each step.
        """
        from app.config import MANIFESTS_DIR
        from app.eeg.io import get_eeg_file_metadata

        root_dir = Path(directory_path).resolve()
        if not root_dir.exists() or not root_dir.is_dir():
            raise ValueError(f"Specified directory does not exist or is not a folder: {directory_path}")

        # Ensure manifests directory exists
        MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
        manifest_file = MANIFESTS_DIR / f"{dataset_id}_manifest.csv"

        # Check existing manifest for resumable execution
        already_processed: Dict[str, Dict[str, Any]] = {}
        if resume and manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        fp = r.get("file_path", "")
                        if fp and r.get("status") == "VALIDATED":
                            already_processed[fp] = r
            except Exception:
                pass

        # Step 1: Detect and parse any clinical metadata / participant tables
        metadata_map = cls._discover_metadata(root_dir)

        # Step 2: Recursively find all supported EEG files
        eeg_files: List[Path] = []
        for ext in SUPPORTED_EXTENSIONS:
            eeg_files.extend(root_dir.rglob(f"*{ext}"))
        eeg_files = sorted(list(set(eeg_files)))

        total_files = len(eeg_files)
        active_ingestion_jobs[dataset_id] = {
            "dataset_id": dataset_id,
            "status": "RUNNING",
            "directory": str(root_dir),
            "total_files": total_files,
            "processed_files": 0,
            "valid_count": 0,
            "exclusions_count": 0,
            "current_file": None,
            "progress_percent": 0.0,
            "errors": [],
            "is_ready_for_training": False
        }

        if total_files == 0:
            active_ingestion_jobs[dataset_id]["status"] = "FAILED"
            active_ingestion_jobs[dataset_id]["message"] = f"No supported EEG files found in {root_dir}."
            return {
                "dataset_id": dataset_id,
                "directory": str(root_dir),
                "total_files_found": 0,
                "valid_participants_count": 0,
                "class_distribution": {},
                "exclusions": [{"reason": f"No supported EEG files ({', '.join(SUPPORTED_EXTENSIONS)}) found in directory."}],
                "is_ready_for_training": False,
                "message": f"No EEG files found in {root_dir}. Please place valid EDF, FIF, MAT, NPY, or CSV files in the folder."
            }

        exclusions: List[Dict[str, Any]] = []
        valid_records: List[Dict[str, Any]] = []
        manifest_rows: List[Dict[str, Any]] = []
        seen_subjects: Set[str] = set()
        class_dist: Dict[str, int] = {}
        sample_rates: Set[float] = set()
        channel_counts: Set[int] = set()

        # Provenance attribution
        provenance_str = "ASEEG Benchmark (Zenodo 18029536)" if "aseeg" in dataset_id.lower() else (
            "B-SNIP 1 Cohort (NIH NDA #2274)" if "bsnip" in dataset_id.lower() else f"Local Authorized Directory ({root_dir.name})"
        )

        manifest_headers = [
            "filename", "file_path", "file_size_bytes", "file_format",
            "subject_id", "diagnostic_label", "sampling_rate", "channels_count",
            "duration_sec", "status", "error_message", "source_dataset", "provenance"
        ]

        # Open manifest CSV in streaming append mode
        with open(manifest_file, "w", newline="", encoding="utf-8") as m_file:
            writer = csv.DictWriter(m_file, fieldnames=manifest_headers)
            writer.writeheader()

            for idx, fpath in enumerate(eeg_files, 1):
                rel_path = str(fpath.relative_to(root_dir)).replace("\\", "/")
                file_size = fpath.stat().st_size
                ext = fpath.suffix.lower()

                active_ingestion_jobs[dataset_id]["current_file"] = rel_path
                active_ingestion_jobs[dataset_id]["processed_files"] = idx
                active_ingestion_jobs[dataset_id]["progress_percent"] = round((idx / total_files) * 100.0, 1)

                # Resumption check: if already processed and validated in previous run
                if rel_path in already_processed:
                    prev = already_processed[rel_path]
                    s_id = prev.get("subject_id")
                    d_lbl = prev.get("diagnostic_label")
                    if s_id and d_lbl and s_id not in seen_subjects:
                        seen_subjects.add(s_id)
                        lbl_key = str(d_lbl)
                        class_dist[lbl_key] = class_dist.get(lbl_key, 0) + 1
                        sr = float(prev.get("sampling_rate", DEFAULT_SAMPLE_RATE))
                        ch_cnt = int(prev.get("channels_count", 16))
                        sample_rates.add(sr)
                        channel_counts.add(ch_cnt)
                        row_data = {
                            "filename": fpath.name,
                            "file_path": rel_path,
                            "file_size_bytes": file_size,
                            "file_format": ext,
                            "subject_id": s_id,
                            "diagnostic_label": d_lbl,
                            "sampling_rate": sr,
                            "channels_count": ch_cnt,
                            "duration_sec": float(prev.get("duration_sec", 0.0)),
                            "status": "VALIDATED",
                            "error_message": "",
                            "source_dataset": dataset_id,
                            "provenance": provenance_str
                        }
                        writer.writerow(row_data)
                        m_file.flush()
                        valid_records.append({
                            "subject_id": s_id,
                            "dataset_id": dataset_id,
                            "diagnostic_label": d_lbl,
                            "age": None,
                            "sex": "U",
                            "site": "Site_1",
                            "recording_path": str(fpath),
                            "file_exists": 1,
                            "is_excluded": 0,
                            "is_synthetic": 0
                        })
                        active_ingestion_jobs[dataset_id]["valid_count"] = len(valid_records)
                        continue

                # Step 3: Resolve participant ID and diagnostic label
                sub_id, diag_label, meta_info = cls._resolve_participant_and_label(fpath, root_dir, metadata_map)

                if not sub_id:
                    err_reason = "Could not associate file with a verified participant identifier in metadata."
                    exclusions.append({"file": rel_path, "subject_id": "UNKNOWN", "reason": err_reason})
                    writer.writerow({
                        "filename": fpath.name, "file_path": rel_path, "file_size_bytes": file_size,
                        "file_format": ext, "subject_id": "UNKNOWN", "diagnostic_label": "UNKNOWN",
                        "sampling_rate": 0, "channels_count": 0, "duration_sec": 0,
                        "status": "EXCLUDED", "error_message": err_reason,
                        "source_dataset": dataset_id, "provenance": provenance_str
                    })
                    m_file.flush()
                    active_ingestion_jobs[dataset_id]["exclusions_count"] = len(exclusions)
                    continue

                if sub_id in seen_subjects:
                    err_reason = f"Duplicate subject ID '{sub_id}' already ingested. Skipping duplicate recording."
                    exclusions.append({"file": rel_path, "subject_id": sub_id, "reason": err_reason})
                    writer.writerow({
                        "filename": fpath.name, "file_path": rel_path, "file_size_bytes": file_size,
                        "file_format": ext, "subject_id": sub_id, "diagnostic_label": diag_label or "UNKNOWN",
                        "sampling_rate": 0, "channels_count": 0, "duration_sec": 0,
                        "status": "DUPLICATE", "error_message": err_reason,
                        "source_dataset": dataset_id, "provenance": provenance_str
                    })
                    m_file.flush()
                    active_ingestion_jobs[dataset_id]["exclusions_count"] = len(exclusions)
                    continue

                if not diag_label or diag_label == "UNKNOWN":
                    err_reason = f"Diagnostic label for participant '{sub_id}' is missing or unverified in metadata."
                    exclusions.append({"file": rel_path, "subject_id": sub_id, "reason": err_reason})
                    writer.writerow({
                        "filename": fpath.name, "file_path": rel_path, "file_size_bytes": file_size,
                        "file_format": ext, "subject_id": sub_id, "diagnostic_label": "UNKNOWN",
                        "sampling_rate": 0, "channels_count": 0, "duration_sec": 0,
                        "status": "EXCLUDED", "error_message": err_reason,
                        "source_dataset": dataset_id, "provenance": provenance_str
                    })
                    m_file.flush()
                    active_ingestion_jobs[dataset_id]["exclusions_count"] = len(exclusions)
                    continue

                # Step 4: Validate file integrity & properties in a chunked/memory-safe manner
                try:
                    meta = get_eeg_file_metadata(str(fpath))
                    srate = meta["sample_rate"]
                    n_chs = meta["channels_count"]
                    dur = meta["duration_sec"]

                    # Quick duration & channel check
                    if dur < 2.0 and dur > 0:
                        raise ValueError(f"Recording duration ({dur}s) is shorter than minimum 2.0s window.")
                    if n_chs < 2:
                        raise ValueError(f"Insufficient EEG channels ({n_chs} found).")

                    sample_rates.add(srate)
                    channel_counts.add(n_chs)

                    # Mark valid
                    seen_subjects.add(sub_id)
                    class_dist[diag_label] = class_dist.get(diag_label, 0) + 1

                    valid_records.append({
                        "subject_id": sub_id,
                        "dataset_id": dataset_id,
                        "diagnostic_label": diag_label,
                        "age": meta_info.get("age"),
                        "sex": meta_info.get("sex", meta_info.get("gender", "U")),
                        "site": meta_info.get("site", "Site_1"),
                        "recording_path": str(fpath),
                        "file_exists": 1,
                        "is_excluded": 0,
                        "is_synthetic": 0
                    })

                    writer.writerow({
                        "filename": fpath.name,
                        "file_path": rel_path,
                        "file_size_bytes": file_size,
                        "file_format": ext,
                        "subject_id": sub_id,
                        "diagnostic_label": diag_label,
                        "sampling_rate": srate,
                        "channels_count": n_chs,
                        "duration_sec": dur,
                        "status": "VALIDATED",
                        "error_message": "",
                        "source_dataset": dataset_id,
                        "provenance": provenance_str
                    })
                    m_file.flush()
                    active_ingestion_jobs[dataset_id]["valid_count"] = len(valid_records)

                except Exception as e:
                    err_msg = str(e)
                    exclusions.append({"file": rel_path, "subject_id": sub_id, "reason": err_msg})
                    writer.writerow({
                        "filename": fpath.name, "file_path": rel_path, "file_size_bytes": file_size,
                        "file_format": ext, "subject_id": sub_id, "diagnostic_label": diag_label,
                        "sampling_rate": 0, "channels_count": 0, "duration_sec": 0,
                        "status": "ERROR", "error_message": err_msg,
                        "source_dataset": dataset_id, "provenance": provenance_str
                    })
                    m_file.flush()
                    active_ingestion_jobs[dataset_id]["exclusions_count"] = len(exclusions)

        # Step 5: Determine task capability and readiness
        scz_count = class_dist.get("SCZ", 0) + class_dist.get("SZ", 0)
        bd_count = class_dist.get("BD", 0)
        hc_count = class_dist.get("HC", 0)

        can_do_scz_vs_bd = (scz_count >= 3 and bd_count >= 3)
        can_do_scz_vs_hc = (scz_count >= 3 and hc_count >= 3)
        is_ready = can_do_scz_vs_bd or can_do_scz_vs_hc

        resolved_task = target_task
        if not resolved_task:
            if can_do_scz_vs_bd:
                resolved_task = "SCZ_VS_BD"
            elif can_do_scz_vs_hc:
                resolved_task = "SZ_VS_HC"
            else:
                resolved_task = "INSUFFICIENT_DATA"

        # Step 6: Store in SQLite database
        d_name = dataset_name or f"Ingested Dataset ({root_dir.name})"
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Register or update dataset
            cursor.execute("""
            INSERT INTO datasets (
                id, name, dataset_type, is_ready, verified_count, missing_count, duplicate_count, excluded_count, notes, updated_at
            ) VALUES (?, ?, ?, ?, ?, 0, 0, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                is_ready = excluded.is_ready,
                verified_count = excluded.verified_count,
                excluded_count = excluded.excluded_count,
                notes = excluded.notes,
                updated_at = CURRENT_TIMESTAMP
            """, (
                dataset_id,
                d_name,
                dataset_id.upper(),
                1 if is_ready else 0,
                len(valid_records),
                len(exclusions),
                f"Directory: {root_dir} | Task: {resolved_task} | Classes: {class_dist}"
            ))

            # Register subjects
            for rec in valid_records:
                cursor.execute("""
                INSERT INTO subjects (
                    subject_id, dataset_id, diagnostic_label, age, sex, site,
                    recording_path, file_exists, is_excluded, is_synthetic
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, 0)
                ON CONFLICT(dataset_id, subject_id) DO UPDATE SET
                    diagnostic_label = excluded.diagnostic_label,
                    age = excluded.age,
                    sex = excluded.sex,
                    site = excluded.site,
                    recording_path = excluded.recording_path,
                    file_exists = 1,
                    is_excluded = 0,
                    is_synthetic = 0
                """, (
                    rec["subject_id"], rec["dataset_id"], rec["diagnostic_label"],
                    rec["age"], rec["sex"], rec["site"], rec["recording_path"]
                ))

        active_ingestion_jobs[dataset_id].update({
            "status": "COMPLETED",
            "progress_percent": 100.0,
            "valid_count": len(valid_records),
            "exclusions_count": len(exclusions),
            "is_ready_for_training": is_ready,
            "manifest_path": str(manifest_file),
            "message": f"Successfully ingested {len(valid_records)} verified participants from {root_dir.name}."
        })

        return {
            "dataset_id": dataset_id,
            "dataset_name": d_name,
            "directory": str(root_dir),
            "total_files_found": len(eeg_files),
            "valid_participants_count": len(valid_records),
            "class_distribution": class_dist,
            "target_task_supported": resolved_task,
            "can_do_scz_vs_bd": can_do_scz_vs_bd,
            "can_do_scz_vs_hc": can_do_scz_vs_hc,
            "detected_sampling_rates": list(sample_rates),
            "detected_channel_counts": list(channel_counts),
            "exclusions_count": len(exclusions),
            "exclusions_sample": exclusions[:20],
            "manifest_path": str(manifest_file),
            "is_ready_for_training": is_ready,
            "message": (
                f"Successfully ingested {len(valid_records)} verified participants from {root_dir.name}."
                if is_ready else
                f"Ingested {len(valid_records)} participants. Insufficient paired classes to train (Need at least 3 participants per class)."
            )
        }

    @classmethod
    def _discover_metadata(cls, root_dir: Path) -> Dict[str, Dict[str, Any]]:
        """
        Searches directory for known metadata files:
        - Excel (.xlsx) like participants info.xlsx or schizophrenia_EI_dynamics_table.xlsx
        - CSV/TSV manifests (participants.tsv, manifest.csv)
        - JSON sidecars
        """
        meta_map: Dict[str, Dict[str, Any]] = {}

        # 1. Search for Excel tables
        for xlsx_file in root_dir.rglob("*.xlsx"):
            try:
                x = pd.ExcelFile(xlsx_file)
                # Check for ASEEG format (Sheet1 = SZ, Sheet2 = HC)
                if set(x.sheet_names) >= {"Sheet1", "Sheet2"}:
                    df_sz = pd.read_excel(xlsx_file, sheet_name="Sheet1")
                    df_hc = pd.read_excel(xlsx_file, sheet_name="Sheet2")
                    for _, r in df_sz.iterrows():
                        sub_num = r.get("Subject ID")
                        if pd.notna(sub_num):
                            age_val = r.get("Age")
                            rec = {
                                "subject_id": f"ASEEG_SZ_{int(sub_num):02d}",
                                "diagnosis": "SZ",
                                "gender": str(r.get("Gender", "U")),
                                "age": float(age_val) if age_val is not None and pd.notna(age_val) else None
                            }
                            meta_map[f"sz{int(sub_num)}"] = rec
                            meta_map[f"s{int(sub_num)}"] = rec
                    for _, r in df_hc.iterrows():
                        sub_num = r.get("Subject ID")
                        if pd.notna(sub_num):
                            age_val = r.get("Age")
                            rec = {
                                "subject_id": f"ASEEG_HC_{int(sub_num):02d}",
                                "diagnosis": "HC",
                                "gender": str(r.get("Gender", "U")),
                                "age": float(age_val) if age_val is not None and pd.notna(age_val) else None
                            }
                            meta_map[f"n{int(sub_num)}"] = rec

                # Check for Zenodo 14808296 format ('Demographic' sheet with code & BNO)
                if "Demographic" in x.sheet_names:
                    df_dem = pd.read_excel(xlsx_file, sheet_name="Demographic")
                    for _, r in df_dem.iterrows():
                        code = str(r.get("code", "")).strip()
                        if code:
                            bno = r.get("BNO")
                            diag = "HC" if pd.isna(bno) else "SCZ"
                            age_val = r.get("age")
                            meta_map[code.lower()] = {
                                "subject_id": code,
                                "diagnosis": diag,
                                "gender": str(r.get("sex", "U")).upper(),
                                "age": float(age_val) if age_val is not None and pd.notna(age_val) else None
                            }
            except Exception:
                pass

        # 2. Search for CSV/TSV manifests
        for tabular_file in list(root_dir.rglob("*.csv")) + list(root_dir.rglob("*.tsv")):
            # Ignore raw signal CSVs (which typically have no 'diagnosis' header)
            try:
                sep = "\t" if tabular_file.suffix == ".tsv" else ","
                sample_df = pd.read_csv(tabular_file, sep=sep, nrows=5)
                cols = [c.lower() for c in sample_df.columns]
                if any(c in cols for c in ["diagnosis", "phenotype", "group", "dx"]):
                    df_full = pd.read_csv(tabular_file, sep=sep)
                    cols_full = {c.lower(): c for c in df_full.columns}
                    id_col = next((cols_full[c] for c in ["subject_id", "participant_id", "subid", "id", "code"] if c in cols_full), None)
                    dx_col = next((cols_full[c] for c in ["diagnosis", "phenotype", "group", "dx", "label"] if c in cols_full), None)
                    if id_col and dx_col:
                        for _, r in df_full.iterrows():
                            s_id = str(r[id_col]).strip()
                            raw_dx = str(r[dx_col]).strip().upper()
                            mapped_dx = "SCZ" if raw_dx in ["SCZ", "SZ", "SCHIZOPHRENIA"] else ("BD" if raw_dx in ["BD", "BIPOLAR", "BP"] else ("HC" if raw_dx in ["HC", "CONTROL", "HEALTHY"] else "OTHER"))
                            age_val = r.get("age")
                            meta_map[s_id.lower()] = {
                                "subject_id": s_id,
                                "diagnosis": mapped_dx,
                                "gender": str(r.get("sex", r.get("gender", "U"))).upper(),
                                "age": float(age_val) if age_val is not None and pd.notna(age_val) else None
                            }
            except Exception:
                pass

        return meta_map

    @classmethod
    def _resolve_participant_and_label(
        cls,
        fpath: Path,
        root_dir: Path,
        metadata_map: Dict[str, Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """
        Associates file with participant metadata based on documented schema conventions.
        """
        stem = fpath.stem.lower()

        # 1. Direct metadata lookup by filename stem
        if stem in metadata_map:
            m = metadata_map[stem]
            return m["subject_id"], m["diagnosis"], m

        # 2. Check parent directory naming (e.g. ASEEG folder convention: 'EO/schizophrenia/s1.mat', 'EO/normal/n1.mat')
        parent_name = fpath.parent.name.lower()
        if (stem.startswith("sz") or stem.startswith("s")) and ("sz" in parent_name or "schizophrenia" in parent_name or stem in metadata_map):
            meta = metadata_map.get(stem, {"age": None, "gender": "U"})
            return f"ASEEG_{stem.upper()}", "SZ", meta
        elif stem.startswith("n") and ("normal" in parent_name or stem in metadata_map):
            meta = metadata_map.get(stem, {"age": None, "gender": "U"})
            return f"ASEEG_{stem.upper()}", "HC", meta

        # 3. Check BIDS participant prefix: sub-xxx
        parts = stem.split("_")
        for p in parts:
            if p.startswith("sub-"):
                clean_id = p.replace("sub-", "").lower()
                if clean_id in metadata_map:
                    m = metadata_map[clean_id]
                    return m["subject_id"], m["diagnosis"], m

        return None, None, {}
