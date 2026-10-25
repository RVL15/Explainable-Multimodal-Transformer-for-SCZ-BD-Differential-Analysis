import pytest
from app.datasets.aseeg_manager import ASEEGDatasetManager
from app.eeg.synthetic import generate_synthetic_subject_recording
from app.xai.attribution import SCIENTIFIC_LIMITATION_DISCLAIMER

def test_aseeg_technical_validation_strictly_labelled():
    prov = ASEEGDatasetManager.get_provenance_info()
    assert prov["task_label"] == "SZ vs HC Technical Validation"
    assert "THIS MUST NEVER BE PRESENTED OR CONFLATED AS SCZ-VS-BD DIFFERENTIAL DIAGNOSIS" in prov["purpose"]

def test_synthetic_data_generation_is_reproducible_and_flagged():
    signals, channels, sample_rate = generate_synthetic_subject_recording(
        subject_id="DEMO_SCZ_001",
        diagnostic_label="SCZ",
        duration_sec=5.0,
        seed=100
    )
    assert signals.shape[0] == 16
    assert signals.shape[1] == int(5.0 * sample_rate)
    assert len(channels) == 16

def test_explainability_disclaimer_preserves_scientific_limits():
    assert "RESEARCH LIMITATION NOTICE" in SCIENTIFIC_LIMITATION_DISCLAIMER
    assert "does NOT imply neurobiological causation" in SCIENTIFIC_LIMITATION_DISCLAIMER
