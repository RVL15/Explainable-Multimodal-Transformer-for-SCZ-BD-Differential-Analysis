from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# ==================== DATASET SCHEMAS ====================
class DatasetSummary(BaseModel):
    id: str
    name: str
    dataset_type: str
    source_url: Optional[str] = None
    is_ready: bool
    verified_count: int
    missing_count: int
    duplicate_count: int
    excluded_count: int
    notes: Optional[str] = None
    updated_at: Optional[str] = None

class SubjectItem(BaseModel):
    id: int
    subject_id: str
    dataset_id: str
    diagnostic_label: str  # 'SCZ', 'BD', 'HC'
    age: Optional[float] = None
    sex: Optional[str] = None
    site: Optional[str] = None
    recording_path: Optional[str] = None
    file_exists: bool
    is_excluded: bool
    exclusion_reason: Optional[str] = None
    is_synthetic: bool

class ManifestValidationResult(BaseModel):
    dataset_id: str
    total_manifest_records: int
    valid_subjects_count: int
    missing_files_count: int
    duplicate_subjects_count: int
    excluded_subjects_count: int
    class_distribution: Dict[str, int]
    exclusions: List[Dict[str, Any]]
    readiness_status: str
    message: str

# ==================== EEG SIGNAL SCHEMAS ====================
class PreprocessingOptions(BaseModel):
    bandpass_enabled: bool = True
    low_freq_hz: float = 1.0
    high_freq_hz: float = 45.0
    rereference_average: bool = True
    normalize_channels: bool = True
    window_duration_sec: float = 2.0
    rejection_threshold_uv: float = 150.0

class ChannelWaveform(BaseModel):
    channel: str
    raw_samples: List[float]
    processed_samples: Optional[List[float]] = None

class SignalExplorerResponse(BaseModel):
    subject_id: str
    dataset_id: str
    diagnostic_label: str
    is_synthetic: bool
    sample_rate: float
    duration_sec: float
    total_time_points: int
    time_vector: List[float]
    channels: List[str]
    waveforms: List[ChannelWaveform]
    total_windows: int
    rejected_windows: int
    rejection_ratio: float
    applied_preprocessing: PreprocessingOptions

class BandPowerItem(BaseModel):
    band: str
    freq_range: str
    absolute_power: float
    relative_power: float

class ChannelSpectralItem(BaseModel):
    channel: str
    frequencies: List[float]
    psd_values: List[float]
    band_powers: List[BandPowerItem]

class SpectralAnalysisResponse(BaseModel):
    subject_id: str
    channels_spectral: List[ChannelSpectralItem]
    mean_band_distribution: Dict[str, float]

class EEGValidationResponse(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    num_channels: int
    num_samples: int
    sample_rate: float
    duration_sec: float
    channels: List[str]
    montage_compliance: str
    quality_score: float
    peak_amplitude_uv: float

class SupportedFormatsResponse(BaseModel):
    supported_extensions: List[str]
    recommended_format: str
    specifications: Dict[str, str]
    multimodal_inputs: Dict[str, str]

# ==================== MACHINE LEARNING & MULTIMODAL SCHEMAS ====================
class TrainModelRequest(BaseModel):
    dataset_id: str = "demo"
    target_task: str = "SCZ_VS_BD"  # 'SCZ_VS_BD' or 'SZ_VS_HC_TECH_VAL'
    model_type: str = "CNN_TRANSFORMER"  # 'CNN_TRANSFORMER' or 'LOGISTIC_REGRESSION'
    epochs: int = 15
    batch_size: int = 16
    learning_rate: float = 0.0005
    transformer_layers: int = 2
    transformer_heads: int = 4
    hidden_dim: int = 64
    dropout: float = 0.2
    test_split_ratio: float = 0.2
    val_split_ratio: float = 0.15
    random_seed: int = 42
    uncertainty_threshold: float = 0.65

class ExperimentSummary(BaseModel):
    id: str
    name: str
    model_type: str
    dataset_id: str
    target_task: str
    status: str
    current_epoch: int
    total_epochs: int
    train_loss_history: List[float]
    val_loss_history: List[float]
    val_metric_history: List[float]
    final_metrics: Optional[Dict[str, Any]] = None
    created_at: str
    finished_at: Optional[str] = None
    error_log: Optional[str] = None

class ModalityEvidenceItem(BaseModel):
    prob_scz: float
    prob_bd: float
    favored_diagnosis: str
    margin: float

class ModalityAgreementInfo(BaseModel):
    status: str  # 'FULL_AGREEMENT', 'PARTIAL_AGREEMENT', 'DISCORDANT_DISAGREEMENT', 'SINGLE_MODALITY'
    concordance_score: float
    description: str

class MultimodalAnalysisRequest(BaseModel):
    subject_id: Optional[str] = None
    uncertainty_threshold: float = 0.65
    mri_features: Optional[Dict[str, float]] = None
    clinical_features: Optional[Dict[str, float]] = None

class MultimodalAnalysisResponse(BaseModel):
    subject_id: Optional[str] = None
    predicted_label: str
    raw_label: str
    prob_scz: float
    prob_bd: float
    confidence: float
    raw_confidence: float
    uncertainty_threshold: float
    is_uncertain: bool
    discordance_penalty: float
    evidence_margin: float
    available_modalities: List[str]
    missing_modalities: List[str]
    modality_evidence: Dict[str, ModalityEvidenceItem]
    modality_agreement: ModalityAgreementInfo
    eeg_channel_attributions: Optional[List[Dict[str, Any]]] = None
    top_contributing_channels: Optional[List[str]] = None
    mri_attributions: Optional[List[Dict[str, Any]]] = None
    clinical_attributions: Optional[List[Dict[str, Any]]] = None
    scientific_disclaimer: str

class ModelStatusResponse(BaseModel):
    models: Dict[str, Any]
    checkpoints: List[Dict[str, Any]]
    active_device: str
    scientific_status: str

# ==================== EVALUATION SCHEMAS ====================
class SubjectPredictionItem(BaseModel):
    subject_id: str
    true_label: str
    predicted_label: str
    prob_scz: float
    prob_bd: float
    is_uncertain: bool
    is_correct: bool

class EvaluationSummary(BaseModel):
    id: str
    experiment_id: str
    dataset_id: str
    target_task: str
    model_type: str
    test_subjects_count: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sensitivity: float
    specificity: float
    roc_auc: Optional[float] = None
    uncertainty_threshold: float
    uncertain_count: int
    coverage: float
    confusion_matrix: List[List[int]]
    subject_predictions: List[SubjectPredictionItem]
    baseline_comparison: Optional[Dict[str, Any]] = None

# ==================== EXPLAINABILITY (XAI) SCHEMAS ====================
class ChannelAttribution(BaseModel):
    channel: str
    importance_score: float
    rank: int

class TimeWindowAttribution(BaseModel):
    window_idx: int
    start_sec: float
    end_sec: float
    attribution_weight: float

class SubjectEvidenceResponse(BaseModel):
    subject_id: str
    dataset_id: str
    true_label: str
    predicted_label: str
    prob_scz: float
    prob_bd: float
    uncertainty_threshold: float
    is_uncertain: bool
    evidence_margin: float
    channel_attributions: List[ChannelAttribution]
    time_attributions: List[TimeWindowAttribution]
    top_contributing_channels: List[str]
    method: str
    scientific_disclaimer: str

# ==================== REPORT SCHEMAS ====================
class ResearchReportOut(BaseModel):
    id: str
    title: str
    experiment_id: Optional[str] = None
    created_at: str
    summary: str
    report_data: Dict[str, Any]
