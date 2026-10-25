import sqlite3
import json
from contextlib import contextmanager
from typing import Generator, Dict, Any, List, Optional
from datetime import datetime
from app.config import DATABASE_PATH

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    """Initializes all necessary SQLite tables for the differential research system."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Datasets table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            dataset_type TEXT NOT NULL, -- 'BSNIP', 'ASEEG_TECH_VAL', 'DEMO'
            source_url TEXT,
            manifest_path TEXT,
            is_ready INTEGER DEFAULT 0,
            verified_count INTEGER DEFAULT 0,
            missing_count INTEGER DEFAULT 0,
            duplicate_count INTEGER DEFAULT 0,
            excluded_count INTEGER DEFAULT 0,
            notes TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Subjects table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            diagnostic_label TEXT NOT NULL, -- 'SCZ', 'BD', 'HC'
            age REAL,
            sex TEXT,
            site TEXT,
            recording_path TEXT,
            file_exists INTEGER DEFAULT 0,
            is_excluded INTEGER DEFAULT 0,
            exclusion_reason TEXT,
            is_synthetic INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(dataset_id, subject_id)
        )
        """)

        # Preprocessing jobs
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS preprocessing_jobs (
            id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            status TEXT NOT NULL, -- 'PENDING', 'PROCESSED', 'FAILED'
            sample_rate REAL,
            channels_count INTEGER,
            duration_sec REAL,
            total_windows INTEGER DEFAULT 0,
            rejected_windows INTEGER DEFAULT 0,
            rejection_ratio REAL DEFAULT 0.0,
            filter_band TEXT,
            rereference TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Experiments table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            model_type TEXT NOT NULL, -- 'CNN_TRANSFORMER', 'LOGISTIC_REGRESSION'
            dataset_id TEXT NOT NULL,
            target_task TEXT NOT NULL, -- 'SCZ_VS_BD', 'SZ_VS_HC_TECH_VAL'
            status TEXT NOT NULL, -- 'IDLE', 'RUNNING', 'COMPLETED', 'FAILED'
            current_epoch INTEGER DEFAULT 0,
            total_epochs INTEGER DEFAULT 0,
            hyperparameters TEXT, -- JSON
            train_loss_history TEXT, -- JSON list
            val_loss_history TEXT, -- JSON list
            val_metric_history TEXT, -- JSON list
            final_metrics TEXT, -- JSON
            checkpoint_path TEXT,
            error_log TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            finished_at TIMESTAMP
        )
        """)

        # Evaluations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            target_task TEXT NOT NULL,
            test_subjects_count INTEGER NOT NULL,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            sensitivity REAL,
            specificity REAL,
            roc_auc REAL,
            uncertainty_threshold REAL NOT NULL,
            uncertain_count INTEGER DEFAULT 0,
            coverage REAL DEFAULT 1.0,
            confusion_matrix TEXT, -- JSON 2x2
            subject_predictions TEXT, -- JSON list
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        )
        """)

        # Research reports
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_reports (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            experiment_id TEXT,
            summary TEXT,
            report_data TEXT, -- JSON
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Analysis history table for tracking live predictions
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            dataset_id TEXT,
            target_task TEXT NOT NULL,
            model_version TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            raw_label TEXT,
            prob_scz REAL,
            prob_bd REAL,
            confidence REAL,
            uncertainty_threshold REAL,
            is_uncertain INTEGER DEFAULT 0,
            discordance_penalty REAL,
            evidence_margin REAL,
            top_channels TEXT, -- JSON list
            attributions_json TEXT, -- JSON full attribution details
            status TEXT DEFAULT 'COMPLETED',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Initialize default dataset rows if not present
        cursor.execute("SELECT COUNT(*) FROM datasets WHERE id = 'bsnip1'")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
            INSERT INTO datasets (id, name, dataset_type, source_url, notes, is_ready)
            VALUES (
                'bsnip1',
                'B-SNIP 1 (Bipolar & Schizophrenia Network on Intermediate Phenotypes)',
                'BSNIP',
                'https://nda.nih.gov/edit_collection.html?id=2274',
                'Access restricted under NIH NDA Data Use Certification. Primary research experiment.',
                0
            )
            """)

        cursor.execute("SELECT COUNT(*) FROM datasets WHERE id = 'aseeg'")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
            INSERT INTO datasets (id, name, dataset_type, source_url, notes, is_ready)
            VALUES (
                'aseeg',
                'ASEEG Technical Validation Dataset (SZ vs HC)',
                'ASEEG_TECH_VAL',
                'https://zenodo.org/records/18029536',
                'Technical validation benchmark (SZ vs Healthy Control). Strictly separated from primary SCZ-vs-BD differential analysis.',
                0
            )
            """)


