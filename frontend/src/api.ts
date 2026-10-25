import {
  Dataset,
  SubjectItem,
  PreprocessingOptions,
  SignalResponse,
  SpectralAnalysis,
  Experiment,
  EvaluationMetrics,
  SubjectEvidence,
  ResearchReport,
  SystemHealth,
  EEGValidationResult,
  SupportedFormats,
  MultimodalAnalysisResult,
  FeaturesSchemaResponse,
  ClassificationTasksReadiness
} from './types';

const API_BASE = '/api';

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorBody.detail || `Request failed with status ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  // Health
  getHealth: (): Promise<SystemHealth> =>
    request<SystemHealth>(`${API_BASE}/health`),

  getClassificationTasksReadiness: (): Promise<ClassificationTasksReadiness> =>
    request<ClassificationTasksReadiness>(`${API_BASE}/health/classification-tasks`),

  // Datasets
  getDatasets: (): Promise<Dataset[]> =>
    request<Dataset[]>(`${API_BASE}/datasets`),

  getDataset: (id: string): Promise<Dataset> =>
    request<Dataset>(`${API_BASE}/datasets/${id}`),

  validateManifest: async (file?: File, csvText?: string, dataDir?: string): Promise<any> => {
    const formData = new FormData();
    if (file) formData.append('file', file);
    if (csvText) formData.append('csv_text', csvText);
    if (dataDir) formData.append('data_directory', dataDir);

    return request<any>(`${API_BASE}/datasets/bsnip/validate-manifest`, {
      method: 'POST',
      body: formData,
    });
  },

  ingestDirectory: (directoryPath: string, datasetId = 'custom', datasetName?: string, targetTask?: string): Promise<any> =>
    request<any>(`${API_BASE}/datasets/ingest-directory`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        directory_path: directoryPath,
        dataset_id: datasetId,
        dataset_name: datasetName,
        target_task: targetTask,
      }),
    }),

  getSubjects: (datasetId: string, label?: string, limit = 100): Promise<{ dataset_id: string; total_count: number; subjects: SubjectItem[] }> => {
    const params = new URLSearchParams();
    if (label) params.append('diagnostic_label', label);
    params.append('limit', limit.toString());
    return request<{ dataset_id: string; total_count: number; subjects: SubjectItem[] }>(
      `${API_BASE}/datasets/${datasetId}/subjects?${params.toString()}`
    );
  },

  getDatasetManifest: (datasetId: string): Promise<any> =>
    request<any>(`${API_BASE}/datasets/${datasetId}/manifest`),

  getDatasetManifestDownloadUrl: (datasetId: string): string =>
    `${API_BASE}/datasets/${datasetId}/manifest/download`,

  getIngestionProgress: (datasetId: string): Promise<any> =>
    request<any>(`${API_BASE}/datasets/${datasetId}/ingestion-progress`),

  // Signals & Validation
  getSupportedFormats: (): Promise<SupportedFormats> =>
    request<SupportedFormats>(`${API_BASE}/signals/supported-formats`),

  validateEegFile: async (file: File): Promise<EEGValidationResult> => {
    const formData = new FormData();
    formData.append('file', file);
    return request<EEGValidationResult>(`${API_BASE}/signals/validate`, {
      method: 'POST',
      body: formData,
    });
  },

  uploadAndPreviewEeg: async (file: File, durationSec = 5.0, maxChannels = 16): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);
    return request<any>(
      `${API_BASE}/signals/upload-preview?duration_sec=${durationSec}&max_channels=${maxChannels}`,
      {
        method: 'POST',
        body: formData,
      }
    );
  },

  getRawSignal: (subjectId: string, durationSec = 5.0, maxChannels = 16): Promise<SignalResponse> =>
    request<SignalResponse>(
      `${API_BASE}/signals/${subjectId}/raw?duration_sec=${durationSec}&max_channels=${maxChannels}`
    ),

  preprocessSignal: (subjectId: string, options: PreprocessingOptions, durationSec = 5.0, maxChannels = 16): Promise<SignalResponse> =>
    request<SignalResponse>(
      `${API_BASE}/signals/${subjectId}/preprocess?duration_sec=${durationSec}&max_channels=${maxChannels}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(options),
      }
    ),

  getSpectral: (subjectId: string, maxChannels = 8): Promise<SpectralAnalysis> =>
    request<SpectralAnalysis>(
      `${API_BASE}/signals/${subjectId}/spectral?max_channels=${maxChannels}`
    ),

  // Training
  startTraining: (config: {
    dataset_id: string;
    target_task: string;
    model_type: string;
    epochs: number;
    batch_size: number;
    learning_rate: number;
    uncertainty_threshold: number;
  }): Promise<{ experiment_id: string; name: string; status: string; message: string }> =>
    request<{ experiment_id: string; name: string; status: string; message: string }>(
      `${API_BASE}/training/start`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      }
    ),

  getExperiments: (): Promise<Experiment[]> =>
    request<Experiment[]>(`${API_BASE}/training/experiments`),

  getExperiment: (id: string): Promise<Experiment> =>
    request<Experiment>(`${API_BASE}/training/experiments/${id}`),

  // Evaluation
  getEvaluation: (experimentId: string): Promise<EvaluationMetrics> =>
    request<EvaluationMetrics>(`${API_BASE}/evaluation/${experimentId}`),

  getLatestEvaluation: (): Promise<{ has_evaluation: boolean; [key: string]: any }> =>
    request<{ has_evaluation: boolean; [key: string]: any }>(`${API_BASE}/evaluation/latest/summary`),

  getVerifiedReport: (): Promise<any> =>
    request<any>(`${API_BASE}/evaluation/verified/report`),

  getExportCsvUrl: (experimentId: string): string =>
    `${API_BASE}/evaluation/${experimentId}/export/csv`,

  getExportJsonUrl: (experimentId: string): string =>
    `${API_BASE}/evaluation/${experimentId}/export/json`,

  // Explainability & Single-Subject Evidence
  getExplainableSubjects: (datasetId = 'aseeg'): Promise<SubjectItem[]> =>
    request<SubjectItem[]>(`${API_BASE}/xai/subjects?dataset_id=${datasetId}`),

  getSubjectEvidence: (subjectId: string, uncertaintyThreshold = 0.65, targetClass?: number): Promise<SubjectEvidence> => {
    const params = new URLSearchParams({ uncertainty_threshold: uncertaintyThreshold.toString() });
    if (targetClass !== undefined) params.append('target_class', targetClass.toString());
    return request<SubjectEvidence>(`${API_BASE}/xai/subject/${subjectId}/evidence?${params.toString()}`);
  },

  // Multimodal Differential Analysis
  getMultimodalFeaturesSchema: (): Promise<FeaturesSchemaResponse> =>
    request<FeaturesSchemaResponse>(`${API_BASE}/multimodal/features-schema`),

  analyzeMultimodal: async (
    params: {
      subjectId?: string;
      uncertaintyThreshold?: number;
      eegFile?: File;
      mriFeatures?: Record<string, number>;
      clinicalFeatures?: Record<string, number>;
      targetTask?: string;
    }
  ): Promise<MultimodalAnalysisResult> => {
    const formData = new FormData();
    if (params.subjectId) formData.append('subject_id', params.subjectId);
    if (params.uncertaintyThreshold !== undefined) {
      formData.append('uncertainty_threshold', params.uncertaintyThreshold.toString());
    }
    if (params.eegFile) formData.append('eeg_file', params.eegFile);
    if (params.targetTask) formData.append('target_task', params.targetTask);
    if (params.mriFeatures) {
      formData.append('mri_features_json', JSON.stringify(params.mriFeatures));
    }
    if (params.clinicalFeatures) {
      formData.append('clinical_features_json', JSON.stringify(params.clinicalFeatures));
    }

    return request<MultimodalAnalysisResult>(`${API_BASE}/multimodal/analyze`, {
      method: 'POST',
      body: formData,
    });
  },

  // Reports
  generateReport: (experimentId?: string): Promise<ResearchReport> => {
    const url = experimentId ? `${API_BASE}/reports/generate?experiment_id=${experimentId}` : `${API_BASE}/reports/generate`;
    return request<ResearchReport>(url, { method: 'POST' });
  },

  getReports: (): Promise<ResearchReport[]> =>
    request<ResearchReport[]>(`${API_BASE}/reports`),

  getReport: (id: string): Promise<ResearchReport> =>
    request<ResearchReport>(`${API_BASE}/reports/${id}`),

  // Activity & History
  getRecentActivity: (limit = 8): Promise<any[]> =>
    request<any[]>(`${API_BASE}/activity/recent?limit=${limit}`),

  getAnalysisHistory: (params?: { search?: string; target_task?: string; limit?: number; offset?: number }): Promise<{ total_count: number; records: any[] }> => {
    const q = new URLSearchParams();
    if (params?.search) q.append('search', params.search);
    if (params?.target_task) q.append('target_task', params.target_task);
    if (params?.limit) q.append('limit', params.limit.toString());
    if (params?.offset) q.append('offset', params.offset.toString());
    return request<{ total_count: number; records: any[] }>(`${API_BASE}/history?${q.toString()}`);
  },

  getHistoryRecord: (id: string): Promise<any> =>
    request<any>(`${API_BASE}/history/${id}`),

  deleteAnalysisRecord: (id: string): Promise<any> =>
    request<any>(`${API_BASE}/history/${id}`, { method: 'DELETE' }),

  getHistoryExportCsvUrl: (): string =>
    `${API_BASE}/history/export/csv`,

  // Sample files for direct demonstration
  getSampleEegFile: async (filename: string): Promise<File> => {
    const res = await fetch(`${API_BASE}/multimodal/sample-file/${filename}`);
    if (!res.ok) throw new Error(`Failed to load sample EEG file: ${filename}`);
    const blob = await res.blob();
    return new File([blob], filename, { type: 'application/octet-stream' });
  },

  // Analysis evidence attribution
  getAnalysisEvidence: (analysisId: string): Promise<any> =>
    request<any>(`${API_BASE}/xai/analysis/${analysisId}`),

  // Demo
  populateDemoCohort: (): Promise<{ status: string; message: string; total_generated: number }> =>
    request<{ status: string; message: string; total_generated: number }>(`${API_BASE}/demo/populate`, { method: 'POST' }),

  getDemoStatus: (): Promise<{ demo_dataset_ready: boolean; synthetic_subjects_count: number; marker: string }> =>
    request<{ demo_dataset_ready: boolean; synthetic_subjects_count: number; marker: string }>(`${API_BASE}/demo/status`),
};
