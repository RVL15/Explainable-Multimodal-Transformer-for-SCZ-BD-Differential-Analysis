import React, { useState, useEffect } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import {
  Brain,
  Activity,
  Layers,
  Clock,
  Info,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  FileSearch,
  Sparkles,
  Sliders,
  ShieldAlert,
  Stethoscope,
  ArrowRight,
  FileText
} from 'lucide-react';
import { api } from '../api';
import { SubjectItem, SubjectEvidence as ISubjectEvidence, AnalysisHistoryRecord } from '../types';

export const Explainability: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const urlAnalysisId = searchParams.get('analysis_id') || '';

  const [historyRecords, setHistoryRecords] = useState<AnalysisHistoryRecord[]>([]);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState<string>(urlAnalysisId);
  const [subjects, setSubjects] = useState<SubjectItem[]>([]);
  const [selectedSubjectId, setSelectedSubjectId] = useState<string>('');
  const [targetClass, setTargetClass] = useState<number | undefined>(undefined);
  const [uncertaintyThreshold, setUncertaintyThreshold] = useState<number>(0.65);

  const [evidence, setEvidence] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Tabs: pure EEG electrode Integrated Gradients vs Multimodal attributions
  const [activeTab, setActiveTab] = useState<'eeg' | 'multimodal'>('eeg');
  const [showTechnicalDetails, setShowTechnicalDetails] = useState<boolean>(false);

  // Load history records and subjects list
  useEffect(() => {
    const initData = async () => {
      try {
        const [hRes, subs] = await Promise.all([
          api.getAnalysisHistory({ limit: 30 }).catch(() => ({ records: [] })),
          api.getExplainableSubjects().catch(() => [])
        ]);
        const records = hRes.records || [];
        setHistoryRecords(records);
        setSubjects(subs || []);

        if (urlAnalysisId) {
          setSelectedAnalysisId(urlAnalysisId);
        } else if (records.length > 0) {
          setSelectedAnalysisId(records[0].id);
        } else if (subs.length > 0) {
          setSelectedSubjectId(subs[0].subject_id);
        }
      } catch (err: any) {
        console.error('Failed to load initial explainability data:', err);
      }
    };
    initData();
  }, [urlAnalysisId]);

  // Fetch attribution whenever selectedAnalysisId or selectedSubjectId changes
  useEffect(() => {
    const fetchAttribution = async () => {
      setLoading(true);
      setError(null);
      try {
        if (selectedAnalysisId) {
          const res = await api.getAnalysisEvidence(selectedAnalysisId);
          setEvidence(res);
        } else if (selectedSubjectId) {
          const res = await api.getSubjectEvidence(selectedSubjectId, uncertaintyThreshold, targetClass);
          setEvidence(res);
        } else {
          setEvidence(null);
        }
      } catch (err: any) {
        setError(err.message || 'Failed to compute Integrated Gradients attribution.');
        setEvidence(null);
      } finally {
        setLoading(false);
      }
    };

    if (selectedAnalysisId || selectedSubjectId) {
      fetchAttribution();
    }
  }, [selectedAnalysisId, selectedSubjectId, uncertaintyThreshold, targetClass]);

  const currentSubject = subjects.find((s) => s.subject_id === selectedSubjectId);

  // Derive plain-English explanation
  const topChannels = evidence?.channel_attributions
    ? [...evidence.channel_attributions].sort((a, b) => b.importance_score - a.importance_score).slice(0, 3)
    : [];

  const topTimeWindow = evidence?.time_attributions
    ? [...evidence.time_attributions].sort((a, b) => b.attribution_weight - a.attribution_weight)[0]
    : null;

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-2xl font-bold tracking-tight text-slate-900">Explainability & Attribution</h1>
            <span className="px-2.5 py-0.5 text-xs font-semibold bg-indigo-50 text-indigo-700 rounded-full border border-indigo-200">
              Integrated Gradients (XAI)
            </span>
          </div>
          <p className="text-sm text-slate-500 mt-1">
            Understand the mathematical evidence and channel/time attributions behind neural network predictions.
          </p>
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center space-x-2 bg-slate-100 p-1 rounded-xl">
          <button
            onClick={() => setActiveTab('eeg')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition flex items-center ${
              activeTab === 'eeg'
                ? 'bg-white text-indigo-700 shadow-2xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Activity className="w-3.5 h-3.5 mr-1.5 text-indigo-600" />
            EEG Saliency & Electrodes
          </button>
          <button
            onClick={() => setActiveTab('multimodal')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition flex items-center ${
              activeTab === 'multimodal'
                ? 'bg-white text-indigo-700 shadow-2xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Layers className="w-3.5 h-3.5 mr-1.5 text-emerald-600" />
            Multimodal Cross-Attention
          </button>
        </div>
      </div>

      {/* Control Panel: Subject selection, threshold, and target class */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
          {/* Analysis / Subject selector */}
          <div className="md:col-span-1">
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
              Select Prediction or Subject:
            </label>
            <select
              value={selectedAnalysisId ? `analysis:${selectedAnalysisId}` : selectedSubjectId ? `subject:${selectedSubjectId}` : ''}
              onChange={(e) => {
                const val = e.target.value;
                if (val.startsWith('analysis:')) {
                  const aid = val.replace('analysis:', '');
                  setSelectedAnalysisId(aid);
                  setSelectedSubjectId('');
                  setSearchParams({ analysis_id: aid });
                } else if (val.startsWith('subject:')) {
                  const sid = val.replace('subject:', '');
                  setSelectedSubjectId(sid);
                  setSelectedAnalysisId('');
                  setSearchParams({ subject_id: sid });
                }
              }}
              className="w-full text-xs bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-900 font-semibold focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition"
            >
              {historyRecords.length > 0 && (
                <optgroup label="Recent Predictions">
                  {historyRecords.map((r) => (
                    <option key={`analysis:${r.id}`} value={`analysis:${r.id}`}>
                      {r.filename} — {r.predicted_label} ({r.created_at?.split(' ')[0] || 'Recent'})
                    </option>
                  ))}
                </optgroup>
              )}
              {subjects.length > 0 && (
                <optgroup label="Benchmark Subjects">
                  {subjects.map((s) => (
                    <option key={`subject:${s.subject_id}`} value={`subject:${s.subject_id}`}>
                      {s.subject_id} — {s.diagnostic_label} ({s.dataset_id})
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          </div>

          {/* Target class for attribution */}
          <div>
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
              Explanation Target Class:
            </label>
            <select
              value={targetClass === undefined ? 'auto' : targetClass}
              onChange={(e) => setTargetClass(e.target.value === 'auto' ? undefined : Number(e.target.value))}
              className="w-full text-xs bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-900 font-semibold focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition"
            >
              <option value="auto">Auto-predicted Class (Recommended)</option>
              <option value="0">Class 0: Schizophrenia (SCZ)</option>
              <option value="1">Class 1: Healthy Control (HC) / Bipolar (BD)</option>
            </select>
          </div>

          {/* Uncertainty threshold slider */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <label className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                Uncertainty Threshold:
              </label>
              <span className="font-mono text-xs font-bold text-indigo-600">
                {(uncertaintyThreshold * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min="0.50"
              max="0.95"
              step="0.05"
              value={uncertaintyThreshold}
              onChange={(e) => setUncertaintyThreshold(parseFloat(e.target.value))}
              className="w-full accent-indigo-600"
            />
          </div>
        </div>

        {/* Selected target quick metadata */}
        {evidence && (
          <div className="pt-3 border-t border-slate-100 flex flex-wrap items-center gap-4 text-xs text-slate-600">
            <div>
              Target: <strong className="text-slate-900 font-mono">{evidence.analysis_id || evidence.subject_id}</strong>
            </div>
            {evidence.filename && (
              <div>
                File: <span className="font-mono text-slate-800">{evidence.filename}</span>
              </div>
            )}
            <div>
              Dataset: <strong className="text-slate-900">{evidence.dataset_id}</strong>
            </div>
            <div>
              Task: <strong className="text-slate-900">{evidence.target_task}</strong>
            </div>
            <div>
              Predicted Diagnosis: <strong className="text-indigo-700">{evidence.predicted_label}</strong>
            </div>
            {evidence.created_at && (
              <div className="text-slate-400 font-mono">
                {evidence.created_at}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-800 flex items-start space-x-3">
          <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <div className="font-bold">Attribution Calculation Failed</div>
            <p className="mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div className="p-12 text-center text-slate-400 text-xs flex flex-col items-center justify-center space-y-2 bg-white rounded-xl border border-slate-200">
          <RefreshCw className="w-5 h-5 animate-spin text-indigo-500" />
          <span>Computing path-integral Integrated Gradients across 50 quadrature steps...</span>
        </div>
      )}

      {/* TAB 1: EEG Channel and Temporal Saliency */}
      {!loading && activeTab === 'eeg' && evidence && (
        <div className="space-y-6">
          {/* Prediction and Confidence Overview Card */}
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-2xs space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-100">
              <div>
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                  Model Prediction for Subject {evidence.subject_id}
                </span>
                <div className="flex items-center space-x-3 mt-1">
                  <span className="text-lg font-bold text-slate-900">
                    {evidence.predicted_label === 'HEALTHY_CONTROL' || evidence.predicted_label === 'HC'
                      ? 'Healthy Control (HC)'
                      : evidence.predicted_label === 'BIPOLAR' || evidence.predicted_label === 'BD'
                      ? 'Bipolar Disorder (BD)'
                      : 'Schizophrenia (SCZ)'}
                  </span>
                  <span className="text-xs px-2 py-0.5 rounded-full font-mono font-semibold bg-slate-100 text-slate-700 border border-slate-200">
                    Confidence: {(evidence.confidence * 100).toFixed(1)}%
                  </span>
                  {evidence.is_uncertain ? (
                    <span className="text-xs px-2 py-0.5 rounded-full font-semibold bg-amber-50 text-amber-700 border border-amber-200">
                      Uncertain Prediction
                    </span>
                  ) : (
                    <span className="text-xs px-2 py-0.5 rounded-full font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                      High Confidence
                    </span>
                  )}
                </div>
              </div>

              <div className="text-right text-xs text-slate-500">
                <div>Dataset: <strong className="text-slate-800">{evidence.dataset_id}</strong></div>
                <div>True Label: <strong className="text-slate-800">{evidence.true_label}</strong></div>
              </div>
            </div>

            {/* Calibrated Probability Bars */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200/80 space-y-2">
                <div className="flex justify-between text-xs font-semibold text-slate-700">
                  <span>Schizophrenia (SCZ) Probability</span>
                  <span className="font-mono font-bold text-indigo-700">
                    {(evidence.prob_scz * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-indigo-600 h-2 rounded-full transition-all"
                    style={{ width: `${evidence.prob_scz * 100}%` }}
                  />
                </div>
              </div>

              <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200/80 space-y-2">
                <div className="flex justify-between text-xs font-semibold text-slate-700">
                  <span>
                    {evidence.dataset_id === 'aseeg' ? 'Healthy Control (HC)' : 'Bipolar Disorder (BD)'} Probability
                  </span>
                  <span className="font-mono font-bold text-emerald-700">
                    {(evidence.prob_bd * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-emerald-500 h-2 rounded-full transition-all"
                    style={{ width: `${evidence.prob_bd * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Plain-English Summary Card */}
          <div className="p-5 bg-indigo-50/70 border border-indigo-200/80 rounded-xl space-y-2">
            <div className="flex items-center space-x-2 text-indigo-900 font-bold text-xs">
              <Sparkles className="w-4 h-4 text-indigo-600 shrink-0" />
              <span>Plain-English Interpretation</span>
            </div>
            <p className="text-xs text-indigo-950 leading-relaxed">
              For this recording, the model’s prediction was most strongly influenced by EEG signals measured at channels{' '}
              <strong>{topChannels.map((c) => c.channel).join(', ')}</strong>. In addition, the time window from{' '}
              <strong>{topTimeWindow?.start_sec}s to {topTimeWindow?.end_sec}s</strong> produced the highest saliency score across the duration of the recording.
            </p>
            <p className="text-[11px] text-indigo-800/80">
              Note: Attribution scores indicate mathematical sensitivity in the neural network's decision surface; they do not indicate a physiological lesion or biological cause.
            </p>
          </div>

          {/* Integrated Gradients Visualizations */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 1. Channel Attribution Bar Chart & Table */}
            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-2xs space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-bold text-sm text-slate-900">Electrode Channel Importance</h3>
                  <p className="text-xs text-slate-500">10–20 standard montage channel attribution scores</p>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-semibold">
                  50-Step Quadrature
                </span>
              </div>

              {/* Channel Ranking List */}
              <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
                {evidence.channel_attributions?.map((ch: any, idx: number) => {
                  const isTop = idx < 3;
                  const pct = Math.min(100, ch.importance_score * 300);
                  return (
                    <div
                      key={ch.channel}
                      className={`p-2.5 rounded-lg border text-xs flex items-center justify-between transition ${
                        isTop
                          ? 'bg-indigo-50/60 border-indigo-200 text-indigo-950 font-semibold'
                          : 'bg-slate-50/70 border-slate-200/80 text-slate-700'
                      }`}
                    >
                      <div className="flex items-center space-x-3">
                        <span className="w-5 font-mono text-[11px] text-slate-400">#{idx + 1}</span>
                        <span className="font-mono font-bold w-10">{ch.channel}</span>
                        {isTop && (
                          <span className="text-[9px] px-1.5 py-0.2 rounded font-bold uppercase bg-indigo-100 text-indigo-800">
                            High Influence
                          </span>
                        )}
                      </div>

                      <div className="flex items-center space-x-3">
                        <div className="w-24 bg-slate-200 rounded-full h-2 overflow-hidden">
                          <div
                            className={`h-2 rounded-full ${isTop ? 'bg-indigo-600' : 'bg-slate-500'}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="font-mono text-[11px] font-semibold w-12 text-right">
                          {(ch.importance_score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* 2. Temporal Window Attribution Saliency */}
            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-2xs space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-bold text-sm text-slate-900">Temporal Window Saliency</h3>
                  <p className="text-xs text-slate-500">Contribution weight of each 2.0-second time interval</p>
                </div>
                <Clock className="w-4 h-4 text-slate-400" />
              </div>

              {/* Time Saliency Timeline */}
              <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
                {evidence.time_attributions?.map((tw: any) => {
                  const pct = Math.min(100, tw.attribution_weight * 300);
                  const isTopWindow = tw.window_idx === topTimeWindow?.window_idx;
                  return (
                    <div
                      key={tw.window_idx}
                      className={`p-2.5 rounded-lg border text-xs flex items-center justify-between transition ${
                        isTopWindow
                          ? 'bg-teal-50/70 border-teal-200 text-teal-950 font-semibold'
                          : 'bg-slate-50/70 border-slate-200/80 text-slate-700'
                      }`}
                    >
                      <div className="flex items-center space-x-2 font-mono">
                        <span className="text-slate-500">W#{tw.window_idx + 1}</span>
                        <span className="text-slate-800">
                          [{tw.start_sec.toFixed(1)}s – {tw.end_sec.toFixed(1)}s]
                        </span>
                        {isTopWindow && (
                          <span className="text-[9px] px-1.5 py-0.2 rounded font-bold uppercase bg-teal-100 text-teal-800">
                            Peak Window
                          </span>
                        )}
                      </div>

                      <div className="flex items-center space-x-3">
                        <div className="w-24 bg-slate-200 rounded-full h-2 overflow-hidden">
                          <div
                            className={`h-2 rounded-full ${isTopWindow ? 'bg-teal-600' : 'bg-slate-400'}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="font-mono text-[11px] font-semibold w-12 text-right">
                          {(tw.attribution_weight * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Technical Details Accordion */}
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs space-y-3">
            <button
              onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
              className="w-full flex items-center justify-between text-xs font-bold text-slate-800"
            >
              <div className="flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-indigo-600" />
                <span>Integrated Gradients Mathematical Configuration</span>
              </div>
              {showTechnicalDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>

            {showTechnicalDetails && (
              <div className="pt-3 border-t border-slate-100 font-mono text-[11px] text-slate-600 space-y-2">
                <div>Quadrature Steps: 50 (Gauss-Legendre Numerical Approximation)</div>
                <div>Baseline Tensor: Zero baseline (μ=0, σ=0 relative to train split normalization)</div>
                <div>Target Class Index: {targetClass === undefined ? 'Auto (predicted logit)' : targetClass}</div>
                <div>Model Architecture: EEGCNNTransformer with Attentive Window Pooling</div>
                <div>Electrode Array: 10-20 Standard (16 Channels: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6)</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Empty State when no prediction or subject is loaded */}
      {!loading && !evidence && (
        <div className="bg-white rounded-xl border border-slate-200 p-12 text-center shadow-2xs space-y-4">
          <div className="w-14 h-14 bg-indigo-50 rounded-2xl flex items-center justify-center mx-auto text-indigo-600">
            <FileSearch className="w-7 h-7" />
          </div>
          <div className="max-w-md mx-auto space-y-1">
            <h3 className="text-base font-bold text-slate-900">No Prediction Attributions Available</h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Upload an authentic multi-channel EEG recording on the Prediction page to compute and inspect live Integrated Gradients channel attributions.
            </p>
          </div>
          <div className="pt-2">
            <Link
              to="/"
              className="inline-flex items-center px-4 py-2 bg-indigo-600 text-white rounded-xl text-xs font-bold hover:bg-indigo-700 transition shadow-xs"
            >
              <Brain className="w-4 h-4 mr-2" />
              Go to Prediction Page
            </Link>
          </div>
        </div>
      )}

      {/* TAB 2: Multimodal Cross-Attention Attribution */}
      {!loading && activeTab === 'multimodal' && evidence && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-2xs space-y-6">
          <div className="space-y-1">
            <h3 className="font-bold text-sm text-slate-900">Multimodal Fusion Cross-Attention Weights</h3>
            <p className="text-xs text-slate-500">
              Demonstrates how the multimodal fusion transformer weighs EEG, structural MRI, and clinical symptom features when making predictions.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
              <div className="flex items-center space-x-2 font-bold text-xs text-slate-800">
                <Activity className="w-4 h-4 text-indigo-600" />
                <span>EEG Modality Encoder</span>
              </div>
              <p className="text-[11px] text-slate-500">
                16-channel EEG spatial-temporal features. Captures high temporal-resolution oscillatory abnormalities (e.g., theta/gamma synchrony).
              </p>
              <div className="text-[10px] font-mono text-slate-600 bg-white p-2 rounded border border-slate-200">
                Weight: Cross-Attention Softmax
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
              <div className="flex items-center space-x-2 font-bold text-xs text-slate-800">
                <Brain className="w-4 h-4 text-emerald-600" />
                <span>Structural MRI Encoder</span>
              </div>
              <p className="text-[11px] text-slate-500">
                Volumetric and cortical thickness measurements (lateral ventricles, hippocampus, superior frontal cortex).
              </p>
              <div className="text-[10px] font-mono text-slate-600 bg-white p-2 rounded border border-slate-200">
                Weight: Cross-Attention Softmax
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
              <div className="flex items-center space-x-2 font-bold text-xs text-slate-800">
                <Stethoscope className="w-4 h-4 text-amber-600" />
                <span>Clinical Symptom Scales</span>
              </div>
              <p className="text-[11px] text-slate-500">
                Phenomenological rating scales (PANSS positive/negative, YMRS mania, MADRS depression, illness duration).
              </p>
              <div className="text-[10px] font-mono text-slate-600 bg-white p-2 rounded border border-slate-200">
                Weight: Cross-Attention Softmax
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Scientific Limitation & Clinical Disclaimer Notice */}
      <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl flex items-start space-x-3 text-xs text-slate-600">
        <ShieldAlert className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-800">Scientific Limitation & Responsible AI Notice: </span>
          Feature attributions computed by Integrated Gradients reflect the sensitivity of the deep learning model’s decision boundary with respect to input perturbations. They do not demonstrate physiological causality, disease pathology, or localize neurological brain lesions. Attribution scores should be interpreted as diagnostic model telemetry for research validation only.
        </div>
      </div>
    </div>
  );
};

export default Explainability;
