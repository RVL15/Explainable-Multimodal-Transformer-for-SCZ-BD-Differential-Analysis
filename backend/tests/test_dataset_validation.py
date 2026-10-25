import pytest
from app.datasets.bsnip_manager import BSNIPDatasetManager

def test_manifest_validation_detects_duplicates_and_missing_files(tmp_path):
    csv_content = """subject_id,diagnosis,recording_file,age,sex
SUB_001,SCZ,non_existent_1.edf,34,M
SUB_002,BD,non_existent_2.edf,42,F
SUB_001,SCZ,non_existent_1.edf,34,M
SUB_003,INVALID_DX,non_existent_3.edf,28,M
"""
    result = BSNIPDatasetManager.validate_manifest_content(csv_content, data_directory=str(tmp_path))
    
    assert result["dataset_id"] == "bsnip1"
    assert result["duplicate_subjects_count"] == 1
    assert result["missing_files_count"] == 2
    assert result["excluded_subjects_count"] >= 3
    assert result["readiness_status"] == "UNAVAILABLE_MISSING_DATA"
    assert "primary cohort is currently UNAVAILABLE" in result["message"]

def test_bsnip_provenance_info_contains_nda_link():
    info = BSNIPDatasetManager.get_provenance_info()
    assert "2274" in info["collection_id"]
    assert "nda.nih.gov" in info["nda_url"]
    assert "NIH NDA Data Use Certification" in info["license"]
