export interface Dataset {
  id: string;
  name: string;
  dataset_type: 'BSNIP' | 'ASEEG_TECH_VAL' | 'DEMO';
  source_url?: string;
  manifest_path?: string;
  is_ready: boolean;
  verified_count: number;
  missing_count: number;
  duplicate_count: number;
  excluded_count: number;
  notes?: string;
  updated_at?: string;
  class_counts?: Record<string, number>;
  provenance?: Record<string, any>;
}

export interface SubjectItem {
  id: number;
  subject_id: string;
  dataset_id: string;
  diagnostic_label: 'SCZ' | 'BD' | 'HC' | 'SZ';
  age?: number;
  sex?: string;
  site?: string;
  recording_path?: string;
  file_exists: boolean;
  is_excluded: boolean;
  exclusion_reason?: string;
  is_synthetic: boolean;
}

export interface PreprocessingOptions {
  bandpass_enabled: boolean;
  low_freq_hz: number;
  high_freq_hz: number;
  rereference_average: boolean;
  normalize_channels: boolean;
  window_duration_sec: number;
  rejection_threshold_uv: number;
}

export interface ChannelWaveform {
  channel: string;
  raw_samples: number[];
  processed_samples?: number[] | null;
}

export interface SignalResponse {
  subject_id: string;
  dataset_id: string;
  diagnostic_label: string;
  is_synthetic: boolean;
  sample_rate: number;
  duration_sec: number;
  total_time_points: number;
  time_vector: number[];
  channels: string[];
  waveforms: ChannelWaveform[];
  total_windows?: number;
  rejected_windows?: number;
  rejection_ratio?: number;
  applied_preprocessing?: PreprocessingOptions;
}

export interface BandPowerItem {
  band: string;
  freq_range: string;
  absolute_power: number;
  relative_power: number;
}

export interface ChannelSpectral {
  channel: string;
  frequencies: number[];
  psd_values: number[];
  band_powers: BandPowerItem[];
}

export interface SpectralAnalysis {
  subject_id: string;
  channels_spectral: ChannelSpectral[];
  mean_band_distribution: Record<string, number>;
}

export interface Experiment {
  id: string;
  name: string;
  model_type: 'CNN_TRANSFORMER' | 'LOGISTIC_REGRESSION';
  dataset_id: string;
  target_task: 'SCZ_VS_BD' | 'SZ_VS_HC_TECH_VAL';
  status: 'IDLE' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  current_epoch: number;
  total_epochs: number;
  train_loss_history: number[];
  val_loss_history: number[];
  val_metric_history: number[];
  final_metrics?: EvaluationMetrics | null;
  hyperparameters?: Record<string, any> | null;
  checkpoint_path?: string;
  error_log?: string;
  created_at: string;
  finished_at?: string;
  evaluation_id?: string;
}

export interface SubjectPrediction {
  subject_id: string;
  true_label: string;
  predicted_label: string;
  prob_scz: number;
  prob_bd: number;
  is_uncertain: boolean;
  is_correct: boolean;
}

export interface EvaluationMetrics {
  id?: string;
  experiment_id: string;
  dataset_id: string;
  target_task: string;
  model_type?: string;
  test_subjects_count: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  sensitivity: number;
  specificity: number;
  roc_auc?: number | null;
  uncertainty_threshold: number;
  uncertain_count: number;
  coverage: number;
  confusion_matrix: number[][];
  subject_predictions: SubjectPrediction[];
  train_loss_history?: number[];
  val_loss_history?: number[];
  baseline_comparison?: {
    baseline_model: string;
    baseline_accuracy: number;
    baseline_f1: number;
    baseline_sensitivity: number;
    baseline_specificity: number;
    delta_accuracy: number;
  } | null;
}

export interface ChannelAttribution {
  channel: string;
  importance_score: number;
  rank: number;
}

export interface TimeAttribution {
  window_idx: number;
  start_sec: number;
  end_sec: number;
  attribution_weight: number;
}

export interface SubjectEvidence {
  subject_id: string;
  dataset_id: string;
  true_label: string;
  predicted_label: string;
  raw_label: string;
  prob_scz: number;
  prob_bd: number;
  confidence: number;
  uncertainty_threshold: number;
  is_uncertain: boolean;
  evidence_margin: number;
  is_synthetic: boolean;
  channel_attributions: ChannelAttribution[];
  time_attributions: TimeAttribution[];
  top_contributing_channels: string[];
  method: string;
  scientific_disclaimer: string;
}

export interface ResearchReport {
  id: string;
  title: string;
  experiment_id?: string;
  summary: string;
  created_at: string;
  report_data: Record<string, any>;
}

export interface SystemHealth {
  status: string;
  system: {
    torch_version: string;
    mne_version: string;
    scipy_version: string;
    sklearn_version: string;
    numpy_version: string;
    compute_device: string;
    cuda_available?: boolean;
  };
  models_status?: {
    eeg_encoder: string;
    mri_encoder: string;
    clinical_encoder: string;
    fusion_transformer: string;
    baseline_model: string;
    checkpoints_available: number;
    checkpoints: Array<{ filename: string; size_bytes: number; modified_at: number }>;
  };
  classification_tasks_readiness?: ClassificationTasksReadiness;
  datasets_status: Dataset[];
  demo_cohort?: {
    ready: boolean;
    synthetic_subjects_count: number;
    label: string;
  };
  supported_input_formats?: string[];
  completed_experiments_count: number;
  preprocessed_jobs_count: number;
  research_notice?: string;
  scientific_integrity_notice?: string;
}

export interface ClassificationTaskInfo {
  task_id: string;
  title: string;
  status: 'MODEL READY' | 'DATA REQUIRED' | 'MODEL NOT TRAINED' | 'STATUS UNKNOWN';
  status_reason?: string;
  explanation?: string;
  dataset: string;
  participants?: number | null;
  participants_verified?: boolean;
  sampling_rate?: string | null;
  channels?: number | null;
  held_out_accuracy?: string | null;
  active_checkpoint?: string | null;
  model_architecture?: string | null;
}

export interface ClassificationTasksReadiness {
  sz_vs_hc: ClassificationTaskInfo;
  scz_vs_bd: ClassificationTaskInfo;
}

export interface EEGValidationResult {
  is_valid: boolean;
  errors: string[];
  warnings: string[];
  num_channels: number;
  num_samples: number;
  sample_rate: number;
  duration_sec: number;
  channels: string[];
  montage_compliance: string;
  quality_score: number;
  peak_amplitude_uv: number;
}

export interface SupportedFormats {
  supported_extensions: string[];
  recommended_format: string;
  specifications: Record<string, string>;
  multimodal_inputs: Record<string, string>;
}

export interface ModalityEvidenceItem {
  prob_scz: number;
  prob_bd: number;
  favored_diagnosis: 'SCZ' | 'BD';
  margin: number;
}

export interface ModalityAgreement {
  status: 'FULL_AGREEMENT' | 'PARTIAL_AGREEMENT' | 'DISCORDANT_DISAGREEMENT' | 'SINGLE_MODALITY';
  concordance_score: number;
  description: string;
}

export interface FeatureAttributionItem {
  feature: string;
  importance_score: number;
  value?: number;
}

export interface MultimodalAnalysisResult {
  analysis_id?: string;
  subject_id?: string;
  true_label?: string;
  predicted_label: string;
  raw_label: string;
  target_task?: string;
  prob_sz?: number;
  prob_hc?: number;
  prob_scz: number;
  prob_bd: number;
  confidence: number;
  raw_confidence: number;
  uncertainty_threshold: number;
  is_uncertain: boolean;
  discordance_penalty: number;
  evidence_margin: number;
  available_modalities: string[];
  missing_modalities: string[];
  modality_evidence: Record<string, ModalityEvidenceItem>;
  modality_agreement: ModalityAgreement;
  eeg_channel_attributions?: ChannelAttribution[];
  top_contributing_channels?: string[];
  mri_attributions?: FeatureAttributionItem[];
  clinical_attributions?: FeatureAttributionItem[];
  scientific_disclaimer: string;
}

export interface FeatureSchemaItem {
  id: string;
  name: string;
  typical_scz?: string;
  typical_bd?: string;
  relevance?: string;
}

export interface FeaturesSchemaResponse {
  mri_roi_features: FeatureSchemaItem[];
  clinical_features: FeatureSchemaItem[];
}

export interface ActivityEvent {
  id: string;
  type: 'TRAINING' | 'EVALUATION' | 'ANALYSIS' | 'DATASET';
  title: string;
  description: string;
  timestamp: string;
  status: string;
}

export interface AnalysisHistoryRecord {
  id: string;
  filename: string;
  dataset_id?: string;
  target_task: string;
  model_version: string;
  predicted_label: string;
  raw_label?: string;
  prob_sz?: number;
  prob_hc?: number;
  prob_scz: number;
  prob_bd: number;
  confidence: number;
  uncertainty_threshold: number;
  is_uncertain: boolean;
  discordance_penalty?: number;
  evidence_margin?: number;
  top_channels: string[];
  status: string;
  created_at: string;
}

