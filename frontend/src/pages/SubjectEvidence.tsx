import React, { useState, useEffect } from 'react';
import {
  FileSearch,
  Sliders,
  ShieldAlert,
  AlertTriangle,
  Sparkles,
  BarChart2,
  Clock,
  Layers,
  Info,
  Upload,
  CheckCircle2,
  XCircle,
  FileText,
  Activity,
  Brain,
  Stethoscope,
  Scale,
  RefreshCw
} from 'lucide-react';
import { api } from '../api';
import {
  SubjectItem,
  SubjectEvidence as ISubjectEvidence,
  MultimodalAnalysisResult,
  EEGValidationResult,
  FeaturesSchemaResponse
} from '../types';
import { StatusBadge } from '../components/StatusBadge';

export const SubjectEvidence: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'multimodal' | 'eeg_saliency'>('multimodal');
  const [subjects, setSubjects] = useState<SubjectItem[]>([]);
  const [selectedSubjectId, setSelectedSubjectId] = useState<string>('');
  const [uncertaintyThreshold, setUncertaintyThreshold] = useState<number>(0.65);
  const [targetClass, setTargetClass] = useState<number | undefined>(undefined);

  // EEG Saliency state
  const [evidence, setEvidence] = useState<ISubjectEvidence | null>(null);
  const [loadingEvidence, setLoadingEvidence] = useState<boolean>(false);

  // Multimodal state
  const [inputMode, setInputMode] = useState<'subject' | 'upload'>('subject');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [validationResult, setValidationResult] = useState<EEGValidationResult | null>(null);
  const [validatingFile, setValidatingFile] = useState<boolean>(false);
  
  // Multimodal modalities toggles
  const [includeMri, setIncludeMri] = useState<boolean>(true);
  const [includeClinical, setIncludeClinical] = useState<boolean>(true);

  // Features schema
  const [schema, setSchema] = useState<FeaturesSchemaResponse | null>(null);

  // Configured MRI & Clinical features
  const [mriValues, setMriValues] = useState<Record<string, number>>({
    left_lateral_ventricle: 1.8,
    right_lateral_ventricle: 1.7,
    left_hippocampus: -1.6,
    right_hippocampus: -1.5,
    left_amygdala: -0.4,
    right_amygdala: -0.3,
    left_thalamus: -1.1,
    right_thalamus: -1.0,
    superior_frontal_thickness: -1.4,
    rostral_middle_frontal_thickness: -1.2,
    inferior_temporal_thickness: -1.0,
    insula_cortical_thickness: -1.1,
    total_gray_matter_volume: -1.3,
    intracranial_volume_norm: 1.0,
  });

  const [clinicalValues, setClinicalValues] = useState<Record<string, number>>({
    panss_positive: 32.0,
    panss_negative: 34.0,
    panss_general: 44.0,
    ymrs_score: 8.0,
    madrs_score: 16.0,
    age_of_onset: 20.0,
    duration_of_illness: 6.0,
    gaf_functioning: 42.0,
  });

  const [multimodalResult, setMultimodalResult] = useState<MultimodalAnalysisResult | null>(null);
  const [runningAnalysis, setRunningAnalysis] = useState<boolean>(false);

  // Load subject list and feature schemas
  useEffect(() => {
    const initData = async () => {
      try {
        const [subs, sch] = await Promise.all([
          api.getExplainableSubjects(),
          api.getMultimodalFeaturesSchema(),
        ]);
        setSubjects(subs);
        if (subs.length > 0) {
          setSelectedSubjectId(subs[0].subject_id);
        }
        setSchema(sch);
      } catch (err) {
        console.error('Error fetching subjects or schema', err);
      }
    };
    initData();
  }, []);

  // Fetch standard EEG saliency when selectedSubjectId or threshold changes
  const loadEegEvidence = async () => {
    if (!selectedSubjectId) return;
    setLoadingEvidence(true);
    try {
      const res = await api.getSubjectEvidence(selectedSubjectId, uncertaintyThreshold, targetClass);
      setEvidence(res);
    } catch (err: any) {
      console.error('Failed to compute EEG attribution', err);
    } finally {
      setLoadingEvidence(false);
    }
  };

  useEffect(() => {
    if (selectedSubjectId) {
      loadEegEvidence();
    }
  }, [selectedSubjectId, uncertaintyThreshold, targetClass]);

  // Handle uploaded EEG file
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadedFile(file);
    setValidatingFile(true);
    try {
      const res = await api.validateEegFile(file);
      setValidationResult(res);
    } catch (err: any) {
      alert(`EEG validation error: ${err.message}`);
      setValidationResult(null);
    } finally {
      setValidatingFile(false);
    }
  };

  // Run Multimodal Transformer Inference
  const handleRunMultimodalAnalysis = async () => {
    setRunningAnalysis(true);
    try {
      const res = await api.analyzeMultimodal({
        subjectId: inputMode === 'subject' ? selectedSubjectId : undefined,
        uncertaintyThreshold,
        eegFile: inputMode === 'upload' ? (uploadedFile || undefined) : undefined,
        mriFeatures: includeMri ? mriValues : undefined,
        clinicalFeatures: includeClinical ? clinicalValues : undefined,
      });
      setMultimodalResult(res);
    } catch (err: any) {
      alert(`Multimodal analysis failed: ${err.message}`);
    } finally {
      setRunningAnalysis(false);
    }
  };

  // Presets for rapid testing
  const applyPreset = (type: 'SCZ_CONCORDANT' | 'BD_CONCORDANT' | 'DISCORDANT') => {
    if (type === 'SCZ_CONCORDANT') {
      setMriValues({
        left_lateral_ventricle: 2.1,
        right_lateral_ventricle: 2.0,
        left_hippocampus: -1.8,
        right_hippocampus: -1.7,
        left_amygdala: -0.6,
        right_amygdala: -0.5,
        left_thalamus: -1.3,
        right_thalamus: -1.2,
        superior_frontal_thickness: -1.6,
        rostral_middle_frontal_thickness: -1.5,
        inferior_temporal_thickness: -1.2,
        insula_cortical_thickness: -1.3,
        total_gray_matter_volume: -1.5,
        intracranial_volume_norm: 1.0,
      });
      setClinicalValues({
        panss_positive: 36.0,
        panss_negative: 38.0,
        panss_general: 48.0,
        ymrs_score: 6.0,
        madrs_score: 18.0,
        age_of_onset: 19.0,
        duration_of_illness: 7.0,
        gaf_functioning: 38.0,
      });
    } else if (type === 'BD_CONCORDANT') {
      setMriValues({
        left_lateral_ventricle: 1.1,
        right_lateral_ventricle: 1.1,
        left_hippocampus: -0.4,
        right_hippocampus: -0.3,
        left_amygdala: 0.6,
        right_amygdala: 0.5,
        left_thalamus: 0.1,
        right_thalamus: 0.2,
        superior_frontal_thickness: -0.4,
        rostral_middle_frontal_thickness: -0.2,
        inferior_temporal_thickness: 0.1,
        insula_cortical_thickness: -0.1,
        total_gray_matter_volume: -0.4,
        intracranial_volume_norm: 1.0,
      });
      setClinicalValues({
        panss_positive: 18.0,
        panss_negative: 12.0,
        panss_general: 28.0,
        ymrs_score: 34.0,
        madrs_score: 22.0,
        age_of_onset: 24.0,
        duration_of_illness: 5.0,
        gaf_functioning: 62.0,
      });
    } else {
      // DISCORDANT: SCZ MRI markers combined with severe Bipolar Mania (YMRS)
      setMriValues({
        left_lateral_ventricle: 2.2,
        right_lateral_ventricle: 2.1,
        left_hippocampus: -2.0,
        right_hippocampus: -1.9,
        left_amygdala: -0.7,
        right_amygdala: -0.6,
        left_thalamus: -1.5,
        right_thalamus: -1.4,
        superior_frontal_thickness: -1.8,
        rostral_middle_frontal_thickness: -1.6,
        inferior_temporal_thickness: -1.3,
        insula_cortical_thickness: -1.4,
        total_gray_matter_volume: -1.7,
        intracranial_volume_norm: 1.0,
      });
      setClinicalValues({
        panss_positive: 14.0,
        panss_negative: 8.0,
        panss_general: 22.0,
        ymrs_score: 42.0,
        madrs_score: 10.0,
        age_of_onset: 26.0,
        duration_of_illness: 3.0,
        gaf_functioning: 58.0,
      });
    }
  };

  const currentSubject = subjects.find((s) => s.subject_id === selectedSubjectId);

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-slate-500 mb-1">
            <FileSearch className="w-4 h-4 text-blue-600" />
            <span>Explainable AI & Multimodal Evidence</span>
          </div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
            Explainable Multimodal Transformer Differential Analysis
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Cross-attention fusion of multi-channel EEG, structural MRI morphometry, and clinical phenotyping with calibrated uncertainty and concordance analysis.
          </p>
        </div>

        {/* Tab switch */}
        <div className="flex bg-slate-200/70 p-1 rounded-xl text-xs font-semibold">
          <button
            onClick={() => setActiveTab('multimodal')}
            className={`px-3 py-1.5 rounded-lg transition ${
              activeTab === 'multimodal'
                ? 'bg-white text-blue-700 shadow-2xs font-bold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Multimodal Fusion & Evidence
          </button>
          <button
            onClick={() => setActiveTab('eeg_saliency')}
            className={`px-3 py-1.5 rounded-lg transition ${
              activeTab === 'eeg_saliency'
                ? 'bg-white text-blue-700 shadow-2xs font-bold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            EEG Sensor Saliency
          </button>
        </div>
      </div>

      {/* Uncertainty Guardrail & Threshold Slider */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center">
              <Sliders className="w-4 h-4 text-blue-600 mr-2" />
              Calibrated Uncertainty Rejection Threshold
            </h2>
            <p className="text-xs text-slate-500 mt-1">
              If confidence falls below this threshold or cross-modal discordance is detected, the model outputs <strong>UNCERTAIN</strong>.
            </p>
          </div>
          <div className="flex items-center space-x-3 shrink-0">
            <span className="text-xs font-mono font-bold text-slate-500">Threshold:</span>
            <span className="text-base font-bold font-mono text-blue-700 bg-blue-50 px-3 py-1 rounded-lg border border-blue-200">
              {uncertaintyThreshold.toFixed(2)}
            </span>
          </div>
        </div>

        <input
          type="range"
          min="0.50"
          max="0.95"
          step="0.05"
          value={uncertaintyThreshold}
          onChange={(e) => setUncertaintyThreshold(parseFloat(e.target.value))}
          className="w-full accent-blue-600 cursor-pointer"
        />

        <div className="flex justify-between text-[11px] text-slate-400 font-mono">
          <span>0.50 (Permissive / Raw Softmax)</span>
          <span>0.65 (Recommended Default)</span>
          <span>0.95 (Ultra-Conservative / High Rejection)</span>
        </div>
      </div>

      {/* TAB 1: MULTIMODAL FUSION & EVIDENCE */}
      {activeTab === 'multimodal' && (
        <div className="space-y-6">
          {/* Input Configuration Box */}
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-100">
              <div>
                <h2 className="text-base font-bold text-slate-900 flex items-center">
                  <Brain className="w-5 h-5 text-indigo-600 mr-2" />
                  Multimodal Input & Modality Selection
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Select database cohort subject or upload standalone EEG recording with optional MRI and Clinical data.
                </p>
              </div>

              {/* Input Mode Selector */}
              <div className="flex items-center space-x-2 bg-slate-100 p-1 rounded-lg text-xs font-semibold">
                <button
                  onClick={() => setInputMode('subject')}
                  className={`px-3 py-1 rounded-md transition ${
                    inputMode === 'subject' ? 'bg-white text-slate-900 shadow-2xs font-bold' : 'text-slate-500'
                  }`}
                >
                  Cohort Subject
                </button>
                <button
                  onClick={() => setInputMode('upload')}
                  className={`px-3 py-1 rounded-md transition ${
                    inputMode === 'upload' ? 'bg-white text-slate-900 shadow-2xs font-bold' : 'text-slate-500'
                  }`}
                >
                  Upload EEG File
                </button>
              </div>
            </div>

            {/* Input Selection Details */}
            {inputMode === 'subject' ? (
              <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                <label className="text-xs font-semibold text-slate-700">Cohort Subject:</label>
                <select
                  value={selectedSubjectId}
                  onChange={(e) => setSelectedSubjectId(e.target.value)}
                  className="text-xs font-mono font-semibold px-3 py-2 rounded-xl border border-slate-200 bg-slate-50 text-slate-800 focus:outline-none focus:border-blue-600 min-w-64"
                >
                  {subjects.map((s) => (
                    <option key={s.subject_id} value={s.subject_id}>
                      {s.subject_id} (True: {s.diagnostic_label}) {s.is_synthetic ? '[DEMO]' : ''}
                    </option>
                  ))}
                </select>
                {currentSubject && <StatusBadge label={currentSubject.diagnostic_label} size="sm" />}
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center space-x-4">
                  <label className="px-4 py-2.5 rounded-xl border-2 border-dashed border-blue-300 hover:border-blue-500 bg-blue-50/40 text-blue-700 text-xs font-bold flex items-center cursor-pointer transition">
                    <Upload className="w-4 h-4 mr-2" />
                    <span>Choose EEG Recording (.npy, .edf, .fif, .mat, .csv)</span>
                    <input type="file" accept=".npy,.edf,.bdf,.fif,.mat,.csv" onChange={handleFileUpload} className="hidden" />
                  </label>
                  {uploadedFile && (
                    <span className="text-xs font-mono font-semibold text-slate-700 bg-slate-100 px-3 py-1.5 rounded-lg border border-slate-200">
                      {uploadedFile.name} ({(uploadedFile.size / 1024).toFixed(1)} KB)
                    </span>
                  )}
                </div>

                {validatingFile && <div className="text-xs text-blue-600 font-semibold animate-pulse">Validating EEG data dimensions and channel count...</div>}

                {validationResult && (
                  <div className={`p-4 rounded-xl border text-xs space-y-1.5 ${validationResult.is_valid ? 'bg-emerald-50 border-emerald-200 text-emerald-900' : 'bg-rose-50 border-rose-200 text-rose-900'}`}>
                    <div className="flex items-center space-x-2 font-bold">
                      {validationResult.is_valid ? <CheckCircle2 className="w-4 h-4 text-emerald-600" /> : <XCircle className="w-4 h-4 text-rose-600" />}
                      <span>Validation Report: {validationResult.is_valid ? 'Valid EEG Recording' : 'Validation Failed'}</span>
                      <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-white/70">Quality Score: {validationResult.quality_score * 100}%</span>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px] text-slate-600 pt-1">
                      <div>Channels: <strong>{validationResult.num_channels}</strong></div>
                      <div>Sample Rate: <strong>{validationResult.sample_rate} Hz</strong></div>
                      <div>Duration: <strong>{validationResult.duration_sec}s</strong></div>
                      <div>Montage: <strong>{validationResult.montage_compliance}</strong></div>
                    </div>
                    {validationResult.warnings.length > 0 && (
                      <div className="text-[11px] text-amber-700 mt-1">
                        <strong>Warnings:</strong> {validationResult.warnings.join('; ')}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Presets and Modality Toggles */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-slate-100">
              {/* Modality Toggles */}
              <div className="space-y-3">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Active Modality Encoders</span>
                <div className="flex flex-wrap gap-3">
                  <label className="flex items-center space-x-2 text-xs font-semibold text-slate-800 bg-slate-50 px-3 py-2 rounded-lg border border-slate-200 cursor-pointer">
                    <input type="checkbox" checked={true} disabled className="accent-blue-600" />
                    <Activity className="w-3.5 h-3.5 text-blue-600" />
                    <span>EEG Encoder (16-Ch CNN-Transf.)</span>
                  </label>

                  <label className="flex items-center space-x-2 text-xs font-semibold text-slate-800 bg-slate-50 px-3 py-2 rounded-lg border border-slate-200 cursor-pointer">
                    <input type="checkbox" checked={includeMri} onChange={(e) => setIncludeMri(e.target.checked)} className="accent-indigo-600" />
                    <Brain className="w-3.5 h-3.5 text-indigo-600" />
                    <span>Structural MRI Encoder</span>
                  </label>

                  <label className="flex items-center space-x-2 text-xs font-semibold text-slate-800 bg-slate-50 px-3 py-2 rounded-lg border border-slate-200 cursor-pointer">
                    <input type="checkbox" checked={includeClinical} onChange={(e) => setIncludeClinical(e.target.checked)} className="accent-emerald-600" />
                    <Stethoscope className="w-3.5 h-3.5 text-emerald-600" />
                    <span>Clinical Scale Encoder</span>
                  </label>
                </div>
              </div>

              {/* Quick Presets */}
              <div className="space-y-3">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Phenotypic Test Presets</span>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => applyPreset('SCZ_CONCORDANT')}
                    className="px-3 py-1.5 rounded-lg bg-rose-50 hover:bg-rose-100 text-rose-700 text-xs font-semibold border border-rose-200 transition"
                  >
                    SCZ Concordant Profile
                  </button>
                  <button
                    onClick={() => applyPreset('BD_CONCORDANT')}
                    className="px-3 py-1.5 rounded-lg bg-blue-50 hover:bg-blue-100 text-blue-700 text-xs font-semibold border border-blue-200 transition"
                  >
                    BD Concordant Profile
                  </button>
                  <button
                    onClick={() => applyPreset('DISCORDANT')}
                    className="px-3 py-1.5 rounded-lg bg-amber-50 hover:bg-amber-100 text-amber-800 text-xs font-semibold border border-amber-300 transition"
                  >
                    Discordant Disagreement
                  </button>
                </div>
              </div>
            </div>

            {/* Run Analysis Action Button */}
            <div className="pt-4 border-t border-slate-100 flex justify-end">
              <button
                onClick={handleRunMultimodalAnalysis}
                disabled={runningAnalysis || (inputMode === 'upload' && !uploadedFile)}
                className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white text-xs font-bold shadow-xs transition flex items-center cursor-pointer"
              >
                {runningAnalysis ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    <span>Running Cross-Modal Transformer...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 mr-2" />
                    <span>Run Explainable Multimodal Analysis</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* ANALYSIS RESULTS CARD */}
          {multimodalResult && (
            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-6">
              {/* Header Status */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-100">
                <div className="space-y-1">
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Multimodal Fused Decision
                  </span>
                  <div className="flex items-center space-x-3">
                    <StatusBadge label={multimodalResult.predicted_label} />
                    <span className="text-xs text-slate-500">
                      Raw Hypothesis: <strong className="text-slate-800">{multimodalResult.raw_label}</strong>
                    </span>
                    <span className="text-xs font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded">
                      Calibrated Confidence: <strong>{(multimodalResult.confidence * 100).toFixed(1)}%</strong>
                    </span>
                  </div>
                </div>

                {/* Modality Agreement Status Badge */}
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-bold text-slate-500">Concordance:</span>
                  <span className={`text-xs font-bold px-3 py-1 rounded-lg border font-mono ${
                    multimodalResult.modality_agreement.status === 'FULL_AGREEMENT'
                      ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                      : multimodalResult.modality_agreement.status === 'PARTIAL_AGREEMENT'
                      ? 'bg-blue-50 text-blue-800 border-blue-200'
                      : multimodalResult.modality_agreement.status === 'DISCORDANT_DISAGREEMENT'
                      ? 'bg-amber-50 text-amber-800 border-amber-300'
                      : 'bg-slate-100 text-slate-700 border-slate-200'
                  }`}>
                    {multimodalResult.modality_agreement.status.replace('_', ' ')}
                  </span>
                </div>
              </div>

              {/* Concordance Description Banner */}
              <div className={`p-4 rounded-xl border text-xs leading-relaxed flex items-start space-x-3 ${
                multimodalResult.modality_agreement.status === 'DISCORDANT_DISAGREEMENT'
                  ? 'bg-amber-50/80 border-amber-300 text-amber-900'
                  : 'bg-slate-50 border-slate-200 text-slate-700'
              }`}>
                {multimodalResult.modality_agreement.status === 'DISCORDANT_DISAGREEMENT' ? (
                  <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
                ) : (
                  <Scale className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
                )}
                <div>
                  <strong className="block font-bold mb-0.5">Modality Agreement Analysis:</strong>
                  {multimodalResult.modality_agreement.description}
                  {multimodalResult.discordance_penalty > 0 && (
                    <span className="block mt-1 text-[11px] font-mono font-bold text-amber-800">
                      Applied Discordance Uncertainty Penalty: -{(multimodalResult.discordance_penalty * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
              </div>

              {/* Fused Posterior Evidence Gauges */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-2">
                  <div className="flex justify-between text-xs font-bold text-slate-800">
                    <span className="text-rose-700">Multimodal SCZ Evidence</span>
                    <span className="font-mono">{(multimodalResult.prob_scz * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-3 rounded-full bg-slate-200 overflow-hidden">
                    <div
                      className="h-full bg-rose-500 rounded-full transition-all duration-500"
                      style={{ width: `${multimodalResult.prob_scz * 100}%` }}
                    />
                  </div>
                  <p className="text-[11px] text-slate-500">Cross-attended posterior probability for Schizophrenia.</p>
                </div>

                <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-2">
                  <div className="flex justify-between text-xs font-bold text-slate-800">
                    <span className="text-blue-700">Multimodal BD Evidence</span>
                    <span className="font-mono">{(multimodalResult.prob_bd * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-3 rounded-full bg-slate-200 overflow-hidden">
                    <div
                      className="h-full bg-blue-600 rounded-full transition-all duration-500"
                      style={{ width: `${multimodalResult.prob_bd * 100}%` }}
                    />
                  </div>
                  <p className="text-[11px] text-slate-500">Cross-attended posterior probability for Bipolar Disorder.</p>
                </div>
              </div>

              {/* Modality-Specific Evidence Breakdown Cards */}
              <div className="space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  Individual Modality Evidence Breakdown
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {Object.entries(multimodalResult.modality_evidence).map(([modName, ev]) => (
                    <div key={modName} className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-xs font-bold font-mono text-slate-800">{modName} Modality</span>
                        <StatusBadge label={ev.favored_diagnosis} size="sm" />
                      </div>
                      <div className="space-y-1 text-[11px] font-mono">
                        <div className="flex justify-between">
                          <span className="text-slate-500">SCZ Evidence:</span>
                          <span className="font-bold text-rose-600">{(ev.prob_scz * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">BD Evidence:</span>
                          <span className="font-bold text-blue-600">{(ev.prob_bd * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between pt-1 border-t border-slate-100">
                          <span className="text-slate-500">Margin:</span>
                          <span className="font-bold text-slate-700">{(ev.margin * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Attributions & Influential Factors */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
                {/* EEG Top Sensors */}
                {multimodalResult.top_contributing_channels && (
                  <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-3">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center">
                      <Activity className="w-3.5 h-3.5 text-blue-600 mr-1.5" />
                      Top EEG Sensors
                    </h4>
                    <div className="flex flex-wrap gap-1.5">
                      {multimodalResult.top_contributing_channels.map((ch, idx) => (
                        <span key={ch} className="px-2 py-1 rounded bg-blue-100 text-blue-800 font-mono text-xs font-bold">
                          #{idx + 1} {ch}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* MRI Influential Regions */}
                {multimodalResult.mri_attributions && multimodalResult.mri_attributions.length > 0 && (
                  <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-3">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center">
                      <Brain className="w-3.5 h-3.5 text-indigo-600 mr-1.5" />
                      Influential MRI Regions
                    </h4>
                    <div className="space-y-1 text-xs">
                      {multimodalResult.mri_attributions.slice(0, 3).map((m) => (
                        <div key={m.feature} className="flex justify-between font-mono text-[11px]">
                          <span className="truncate max-w-36 text-slate-700">{m.feature.replace(/_/g, ' ')}</span>
                          <span className="font-bold text-indigo-700">{(m.importance_score * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Clinical Influential Scales */}
                {multimodalResult.clinical_attributions && multimodalResult.clinical_attributions.length > 0 && (
                  <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-3">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center">
                      <Stethoscope className="w-3.5 h-3.5 text-emerald-600 mr-1.5" />
                      Influential Clinical Scales
                    </h4>
                    <div className="space-y-1 text-xs">
                      {multimodalResult.clinical_attributions.slice(0, 3).map((c) => (
                        <div key={c.feature} className="flex justify-between font-mono text-[11px]">
                          <span className="truncate max-w-36 text-slate-700">{c.feature.replace(/_/g, ' ')}</span>
                          <span className="font-bold text-emerald-700">{(c.importance_score * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: EEG SENSOR & TEMPORAL SALIENCY (Path-Integral Integrated Gradients) */}
      {activeTab === 'eeg_saliency' && evidence && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-100">
              <div className="space-y-1">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Selected Subject EEG Evidence
                </span>
                <div className="flex items-center space-x-3">
                  <StatusBadge label={evidence.predicted_label} />
                  <span className="text-xs text-slate-500">
                    True Label: <strong className="text-slate-800">{evidence.true_label}</strong>
                  </span>
                  {evidence.is_synthetic && (
                    <span className="text-xs text-purple-700 font-mono font-bold bg-purple-50 px-2 py-0.5 rounded border border-purple-200">
                      [SYNTHETIC DEMO]
                    </span>
                  )}
                </div>
              </div>

              {evidence.is_uncertain && (
                <div className="px-4 py-2 rounded-lg bg-amber-50 border border-amber-300 text-xs text-amber-900 font-medium flex items-center">
                  <AlertTriangle className="w-4 h-4 text-amber-600 mr-2 shrink-0" />
                  Evidence confidence ({(evidence.confidence * 100).toFixed(1)}%) is below the {evidence.uncertainty_threshold} threshold. Flagged as UNCERTAIN.
                </div>
              )}
            </div>

            {/* Gauges */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-2">
                <div className="flex justify-between text-xs font-bold text-slate-800">
                  <span className="text-rose-700">Schizophrenia (SCZ) Evidence</span>
                  <span className="font-mono">{(evidence.prob_scz * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-3 rounded-full bg-slate-200 overflow-hidden">
                  <div className="h-full bg-rose-500 rounded-full transition-all duration-500" style={{ width: `${evidence.prob_scz * 100}%` }} />
                </div>
                <p className="text-[11px] text-slate-500">Posterior probability supporting schizophrenia classification hypothesis.</p>
              </div>

              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-2">
                <div className="flex justify-between text-xs font-bold text-slate-800">
                  <span className="text-blue-700">Bipolar Disorder (BD) Evidence</span>
                  <span className="font-mono">{(evidence.prob_bd * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-3 rounded-full bg-slate-200 overflow-hidden">
                  <div className="h-full bg-blue-600 rounded-full transition-all duration-500" style={{ width: `${evidence.prob_bd * 100}%` }} />
                </div>
                <p className="text-[11px] text-slate-500">Posterior probability supporting bipolar disorder classification hypothesis.</p>
              </div>
            </div>
          </div>

          {/* Integrated Gradients Rankings */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Channel Ranking */}
            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800">
                  Electrode Channel Importance Ranking
                </h2>
                <span className="text-xs font-mono text-slate-400">Integrated Gradients</span>
              </div>
              <p className="text-xs text-slate-500">
                Mean absolute gradient attribution with respect to the predicted class across 10-20 electrode positions.
              </p>
              <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
                {evidence.channel_attributions.map((ch) => {
                  const isTop = ch.rank <= 3;
                  return (
                    <div
                      key={ch.channel}
                      className={`p-2.5 rounded-lg border text-xs flex items-center justify-between transition ${
                        isTop ? 'bg-blue-50/70 border-blue-200 text-blue-900 font-semibold' : 'bg-slate-50 border-slate-200 text-slate-700'
                      }`}
                    >
                      <div className="flex items-center space-x-3">
                        <span className="w-6 font-mono text-[11px] text-slate-400">#{ch.rank}</span>
                        <span className="font-mono font-bold">{ch.channel}</span>
                        {isTop && (
                          <span className="text-[10px] bg-blue-200 text-blue-800 px-1.5 py-0.2 rounded font-mono font-bold">
                            TOP SENSOR
                          </span>
                        )}
                      </div>
                      <div className="flex items-center space-x-3">
                        <div className="w-24 h-2 bg-slate-200 rounded-full overflow-hidden">
                          <div className="h-full bg-blue-600 rounded-full" style={{ width: `${Math.min(100, ch.importance_score * 300)}%` }} />
                        </div>
                        <span className="font-mono text-[11px] text-slate-600 w-12 text-right">
                          {(ch.importance_score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Time Window Saliency */}
            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800">
                  Temporal Window Attribution Saliency
                </h2>
                <Clock className="w-4 h-4 text-slate-400" />
              </div>
              <p className="text-xs text-slate-500">
                Contribution weight of each 2.0-second time window across the recording duration.
              </p>
              <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
                {evidence.time_attributions.map((tw) => (
                  <div
                    key={tw.window_idx}
                    className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs flex items-center justify-between"
                  >
                    <div className="flex items-center space-x-2 font-mono">
                      <span className="text-slate-500">Window #{tw.window_idx + 1}</span>
                      <span className="text-slate-700">({tw.start_sec}s - {tw.end_sec}s)</span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="w-28 h-2 bg-slate-200 rounded-full overflow-hidden">
                        <div className="h-full bg-teal-600 rounded-full" style={{ width: `${Math.min(100, tw.attribution_weight * 300)}%` }} />
                      </div>
                      <span className="font-mono text-[11px] text-slate-700 font-bold w-12 text-right">
                        {(tw.attribution_weight * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Technical Attribution Note */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 flex items-start space-x-3 text-xs text-slate-500">
        <Info className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-700">Attribution Methodology: </span>
          Integrated Gradients computes path integrals along straightline interpolations between baseline zero-signals and empirical inputs to evaluate model sensitivity across 10-20 electrode montages.
        </div>
      </div>
    </div>
  );
};

export default SubjectEvidence;
