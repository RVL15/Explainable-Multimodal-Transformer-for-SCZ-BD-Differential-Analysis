import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Upload,
  Brain,
  CheckCircle2,
  AlertOctagon,
  AlertCircle,
  FileText,
  Clock,
  Layers,
  Sparkles,
  Download,
  Eye,
  RefreshCw,
  Sliders,
  ShieldCheck,
  ChevronRight,
  Info
} from 'lucide-react';
import { api } from '../api';
import { EEGValidationResult, MultimodalAnalysisResult, ClassificationTasksReadiness } from '../types';
import { StatusBadge } from '../components/StatusBadge';

export const AnalyzeEEG: React.FC = () => {
  const navigate = useNavigate();

  // Task & System State
  const [tasksReadiness, setTasksReadiness] = useState<ClassificationTasksReadiness | null>(null);
  const [selectedTask, setSelectedTask] = useState<'SZ_VS_HC' | 'SCZ_VS_BD'>('SZ_VS_HC');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [validationResult, setValidationResult] = useState<EEGValidationResult | null>(null);
  const [validating, setValidating] = useState<boolean>(false);
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [loadingSample, setLoadingSample] = useState<boolean>(false);
  const [uncertaintyThreshold, setUncertaintyThreshold] = useState<number>(0.65);
  const [analysisResult, setAnalysisResult] = useState<MultimodalAnalysisResult | null>(null);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState<boolean>(false);

  useEffect(() => {
    const loadTasks = async () => {
      try {
        const res = await api.getClassificationTasksReadiness();
        setTasksReadiness(res);
      } catch (err) {
        console.error('Failed to load tasks readiness', err);
      }
    };
    loadTasks();
  }, []);

  // File drag & drop handlers
  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processSelectedFile(e.target.files[0]);
    }
  };

  const processSelectedFile = async (file: File) => {
    setUploadedFile(file);
    setAnalysisResult(null);
    setValidating(true);
    try {
      const res = await api.validateEegFile(file);
      setValidationResult(res);
    } catch (err: any) {
      alert(`EEG validation error: ${err.message}`);
      setValidationResult(null);
    } finally {
      setValidating(false);
    }
  };

  // Load genuine sample recording for viva demonstration
  const handleLoadSample = async (filename: string) => {
    setLoadingSample(true);
    try {
      const file = await api.getSampleEegFile(filename);
      await processSelectedFile(file);
    } catch (err: any) {
      alert(`Could not load genuine sample ${filename}: ${err.message}`);
    } finally {
      setLoadingSample(false);
    }
  };

  // Run Real Prediction
  const handleRunAnalysis = async () => {
    if (!uploadedFile) {
      alert('Please upload an EEG file before analyzing.');
      return;
    }
    setAnalyzing(true);
    try {
      const res = await api.analyzeMultimodal({
        eegFile: uploadedFile,
        uncertaintyThreshold: uncertaintyThreshold,
        targetTask: selectedTask
      });
      setAnalysisResult(res);
    } catch (err: any) {
      alert(`Inference failed: ${err.message}`);
    } finally {
      setAnalyzing(false);
    }
  };

  // Export report as JSON
  const handleExportReport = () => {
    if (!analysisResult) return;
    const reportBlob = new Blob([JSON.stringify(analysisResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(reportBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `analysis_report_${uploadedFile?.name || 'eeg'}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const szHcStatus = tasksReadiness?.sz_vs_hc?.status || 'MODEL READY';
  const sczBdStatus = tasksReadiness?.scz_vs_bd?.status || 'DATA REQUIRED';
  const isSczBdAvailable = sczBdStatus === 'MODEL READY';

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-8">
      {/* Title Header */}
      <div>
        <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-slate-500 mb-1">
          <Brain className="w-4 h-4 text-indigo-600" />
          <span>Neuro-Oscillatory Prediction System</span>
        </div>
        <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
          Explainable Multimodal Transformer for SCZ–BD Differential Analysis
        </h1>
        <p className="text-xs text-slate-500 mt-1 leading-relaxed">
          Upload an authentic multi-channel EEG recording, run automated neurophysiological validation, and receive calibrated deep learning predictions with Integrated Gradients electrode attribution rankings.
        </p>
      </div>

      {/* Step 1: Select Classification Task */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <div className="flex items-center space-x-2 text-xs font-bold text-slate-500 uppercase tracking-wider">
          <span className="w-5 h-5 rounded-full bg-indigo-100 text-indigo-700 flex items-center justify-center font-bold">1</span>
          <span>Select Classification Task</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Option A: SZ vs HC (Active Trained Model) */}
          <div
            onClick={() => setSelectedTask('SZ_VS_HC')}
            className={`p-4 rounded-xl border cursor-pointer transition-all ${
              selectedTask === 'SZ_VS_HC'
                ? 'border-indigo-600 bg-indigo-50/40 ring-2 ring-indigo-500/20'
                : 'border-slate-200 hover:border-slate-300 bg-white'
            }`}
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-bold text-slate-900">1. Schizophrenia vs. Healthy Controls</span>
              <StatusBadge label={szHcStatus} size="sm" />
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">
              Trained on 97 verified ASEEG benchmark participants (500 Hz, 19-Ch). 80.0% held-out test accuracy.
            </p>
            <div className="mt-3 text-[11px] text-indigo-700 font-semibold flex items-center">
              Active Checkpoint: best_model.pt (PyTorch CNN–Transformer)
            </div>
          </div>

          {/* Option B: SCZ vs BD (Primary Differential) */}
          <div
            onClick={() => {
              if (isSczBdAvailable) {
                setSelectedTask('SCZ_VS_BD');
              }
            }}
            className={`p-4 rounded-xl border transition-all ${
              isSczBdAvailable
                ? selectedTask === 'SCZ_VS_BD'
                  ? 'border-indigo-600 bg-indigo-50/40 ring-2 ring-indigo-500/20 cursor-pointer'
                  : 'border-slate-200 hover:border-slate-300 bg-white cursor-pointer'
                : 'border-slate-200 bg-slate-50/60 opacity-80 cursor-not-allowed'
            }`}
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-bold text-slate-900">2. Schizophrenia vs. Bipolar Disorder</span>
              <StatusBadge label={sczBdStatus} size="sm" />
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">
              Primary clinical differential task.
            </p>
            <div className="mt-3 p-2 bg-amber-50/80 rounded-lg border border-amber-200/80 text-[11px] text-amber-900 leading-snug">
              SCZ vs. BD prediction is currently unavailable because the required authorized dataset or trained model is not ready.
            </div>
          </div>
        </div>
      </div>

      {/* Step 2: Upload EEG Recording */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <div className="flex items-center space-x-2 text-xs font-bold text-slate-500 uppercase tracking-wider">
          <span className="w-5 h-5 rounded-full bg-indigo-100 text-indigo-700 flex items-center justify-center font-bold">2</span>
          <span>Upload EEG Recording File</span>
        </div>

        {/* Drag & Drop Upload Zone */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleFileDrop}
          className="border-2 border-dashed border-slate-300 hover:border-indigo-500 rounded-xl p-8 text-center bg-slate-50/50 hover:bg-indigo-50/20 transition cursor-pointer"
        >
          <Upload className="w-8 h-8 text-indigo-600 mx-auto mb-2" />
          <h3 className="text-sm font-bold text-slate-800">
            Drag and drop your EEG file here, or click to browse
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Supported formats: <strong>.mat</strong> (MATLAB v5/v7 & v7.3 HDF5), <strong>.edf</strong>, <strong>.bdf</strong>, <strong>.fif</strong>, <strong>.npy</strong>, <strong>.csv</strong>
          </p>
          <input
            type="file"
            id="eeg-file-input"
            accept=".mat,.edf,.bdf,.fif,.npy,.csv"
            onChange={handleFileInputChange}
            className="hidden"
          />
          <div className="mt-4 flex flex-col items-center justify-center space-y-3">
            <label
              htmlFor="eeg-file-input"
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-xs font-bold hover:bg-indigo-700 transition cursor-pointer shadow-xs"
            >
              Select File from Disk
            </label>

            {/* Quick Demo Sample Files for Viva Testing */}
            <div className="flex flex-wrap items-center justify-center gap-2 pt-2 border-t border-slate-200/80 text-xs">
              <span className="text-slate-500 font-medium text-[11px]">Quick Viva Test Samples:</span>
              <button
                type="button"
                disabled={loadingSample}
                onClick={(e) => {
                  e.stopPropagation();
                  handleLoadSample('sz1.mat');
                }}
                className="px-2.5 py-1 bg-rose-50 border border-rose-200 text-rose-700 hover:bg-rose-100 rounded-md font-mono text-[11px] font-semibold transition"
              >
                sz1.mat (Schizophrenia)
              </button>
              <button
                type="button"
                disabled={loadingSample}
                onClick={(e) => {
                  e.stopPropagation();
                  handleLoadSample('n1.mat');
                }}
                className="px-2.5 py-1 bg-emerald-50 border border-emerald-200 text-emerald-700 hover:bg-emerald-100 rounded-md font-mono text-[11px] font-semibold transition"
              >
                n1.mat (Healthy Control)
              </button>
            </div>
          </div>
        </div>

        {/* Uploaded File & Validation Summary */}
        {validating && (
          <div className="p-4 bg-slate-50 rounded-lg text-xs text-slate-600 flex items-center space-x-2">
            <RefreshCw className="w-4 h-4 text-indigo-600 animate-spin" />
            <span>Validating EEG channel montage, sampling rate, and voltage integrity...</span>
          </div>
        )}

        {uploadedFile && validationResult && (
          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <FileText className="w-4 h-4 text-indigo-600" />
                <span className="font-mono text-xs font-bold text-slate-900">{uploadedFile.name}</span>
                <span className="text-[11px] text-slate-500 font-mono">
                  ({(uploadedFile.size / 1024 / 1024).toFixed(2)} MB)
                </span>
              </div>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                validationResult.is_valid ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'
              }`}>
                {validationResult.is_valid ? 'VALIDATED' : 'VALIDATION WARNING'}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-2.5 bg-white rounded-lg border border-slate-200">
                <span className="text-slate-400 text-[10px] block">Channels Detected</span>
                <strong className="text-slate-900">{validationResult.num_channels} Electrodes</strong>
              </div>
              <div className="p-2.5 bg-white rounded-lg border border-slate-200">
                <span className="text-slate-400 text-[10px] block">Sampling Frequency</span>
                <strong className="text-slate-900">{validationResult.sample_rate} Hz</strong>
              </div>
              <div className="p-2.5 bg-white rounded-lg border border-slate-200">
                <span className="text-slate-400 text-[10px] block">Duration</span>
                <strong className="text-slate-900">{validationResult.duration_sec.toFixed(1)} Seconds</strong>
              </div>
              <div className="p-2.5 bg-white rounded-lg border border-slate-200">
                <span className="text-slate-400 text-[10px] block">Montage Standard</span>
                <strong className="text-emerald-700">{validationResult.montage_compliance}</strong>
              </div>
            </div>

            {validationResult.warnings && validationResult.warnings.length > 0 && (
              <div className="text-[11px] text-amber-800 bg-amber-50 p-2.5 rounded-lg border border-amber-200">
                <strong>Notice: </strong> {validationResult.warnings.join(' • ')}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Step 3: Preprocessing Pipeline Confirmation */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <div className="flex items-center space-x-2 text-xs font-bold text-slate-500 uppercase tracking-wider">
          <span className="w-5 h-5 rounded-full bg-indigo-100 text-indigo-700 flex items-center justify-center font-bold">3</span>
          <span>Review Preprocessing Pipeline</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 text-xs">
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
            <span className="text-slate-400 text-[10px] block uppercase font-bold">1. Channel Alignment</span>
            <span className="text-slate-800 font-semibold text-[11px] block mt-1">16 Standard 10–20</span>
            <span className="text-[10px] text-slate-400">Maps Fp1..T6 to model schema</span>
          </div>
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
            <span className="text-slate-400 text-[10px] block uppercase font-bold">2. Bandpass Filter</span>
            <span className="text-slate-800 font-semibold text-[11px] block mt-1">1.0 – 45.0 Hz</span>
            <span className="text-[10px] text-slate-400">Butterworth 4th-order zero-phase</span>
          </div>
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
            <span className="text-slate-400 text-[10px] block uppercase font-bold">3. Rereferencing</span>
            <span className="text-slate-800 font-semibold text-[11px] block mt-1">Common Average (CAR)</span>
            <span className="text-[10px] text-slate-400">Removes reference electrode bias</span>
          </div>
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
            <span className="text-slate-400 text-[10px] block uppercase font-bold">4. Normalization</span>
            <span className="text-slate-800 font-semibold text-[11px] block mt-1">Learned Statistics</span>
            <span className="text-[10px] text-slate-400">Normalized via training parameters</span>
          </div>
        </div>

        {/* Step 4: Run Analysis Button */}
        <div className="pt-2 flex items-center justify-between">
          <div className="flex items-center space-x-2 text-xs">
            <span className="text-slate-600 font-semibold">Uncertainty Threshold:</span>
            <input
              type="range"
              min="0.50"
              max="0.85"
              step="0.05"
              value={uncertaintyThreshold}
              onChange={(e) => setUncertaintyThreshold(parseFloat(e.target.value))}
              className="w-28 accent-indigo-600 cursor-pointer"
            />
            <span className="font-mono text-slate-800 font-bold">{uncertaintyThreshold}</span>
          </div>

          <button
            onClick={handleRunAnalysis}
            disabled={!uploadedFile || analyzing}
            className={`px-6 py-2.5 rounded-xl text-xs font-bold flex items-center shadow-xs transition ${
              !uploadedFile || analyzing
                ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                : 'bg-indigo-600 text-white hover:bg-indigo-700'
            }`}
          >
            {analyzing ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                Executing PyTorch Inference...
              </>
            ) : (
              <>
                <Brain className="w-4 h-4 mr-2" />
                Analyze EEG Recording
              </>
            )}
          </button>
        </div>
      </div>

      {/* Step 5: Prominent Prediction Results Card */}
      {analysisResult && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-6">
          <div className="flex items-center justify-between border-b border-slate-100 pb-4">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-700 block mb-0.5">
                Model Inference Output
              </span>
              <h2 className="text-xl font-extrabold text-slate-900 tracking-tight">
                Classification Result: {analysisResult.predicted_label}
              </h2>
            </div>
            <div className="flex items-center space-x-2">
              <span className={`text-xs font-bold px-3 py-1 rounded-full ${
                analysisResult.is_uncertain
                  ? 'bg-amber-100 text-amber-800'
                  : 'bg-emerald-100 text-emerald-800'
              }`}>
                {analysisResult.is_uncertain ? 'UNCERTAIN PREDICTION' : 'CONFIDENT PREDICTION'}
              </span>
            </div>
          </div>

          {/* Plain English Prediction Summary */}
          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 text-xs text-slate-700 space-y-2">
            <div className="flex items-start space-x-2">
              <Info className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
              <div>
                <strong>Interpretation: </strong>
                {analysisResult.predicted_label === 'SZ' || analysisResult.predicted_label === 'SCZ' ? (
                  <span>
                    The model identified neuro-oscillatory signatures consistent with the <strong>Schizophrenia (SZ)</strong> class based on temporal convolutional and self-attention patterns.
                  </span>
                ) : analysisResult.predicted_label === 'HC' ? (
                  <span>
                    The model identified normative oscillatory rhythms consistent with the <strong>Healthy Control (HC)</strong> class.
                  </span>
                ) : analysisResult.predicted_label === 'BD' ? (
                  <span>
                    The model assigned this recording to the <strong>Bipolar Disorder (BD)</strong> class.
                  </span>
                ) : (
                  <span>
                    The model confidence fell below the calibrated uncertainty threshold ({uncertaintyThreshold}). The result is classified as <strong>UNCERTAIN</strong> rather than forcing an ambiguous prediction.
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Probability & Evidence Margin */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-2">
              <span className="text-slate-500 font-semibold block">Model Probability Distribution</span>
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between text-[11px] mb-1">
                    <span className="font-semibold text-slate-700">Schizophrenia (SZ)</span>
                    <span className="font-mono font-bold text-indigo-700">{(analysisResult.prob_scz * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                    <div className="h-full bg-indigo-600 rounded-full" style={{ width: `${analysisResult.prob_scz * 100}%` }} />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-[11px] mb-1">
                    <span className="font-semibold text-slate-700">
                      {selectedTask === 'SZ_VS_HC' ? 'Healthy Control (HC)' : 'Bipolar Disorder (BD)'}
                    </span>
                    <span className="font-mono font-bold text-emerald-700">{(analysisResult.prob_bd * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                    <div className="h-full bg-emerald-600 rounded-full" style={{ width: `${analysisResult.prob_bd * 100}%` }} />
                  </div>
                </div>
              </div>
            </div>

            <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-2">
              <span className="text-slate-500 font-semibold block">Top Contributing Electrodes (Attribution)</span>
              <div className="flex flex-wrap gap-1.5 pt-1">
                {analysisResult.top_contributing_channels?.slice(0, 8).map((ch, idx) => (
                  <span
                    key={ch}
                    className="px-2 py-1 bg-white rounded-md border border-slate-200 font-mono text-[11px] font-bold text-slate-800 shadow-2xs"
                  >
                    #{idx + 1} {ch}
                  </span>
                ))}
              </div>
              <p className="text-[10px] text-slate-400 pt-1">
                Calculated via Integrated Gradients path integrals across 16 channels.
              </p>
            </div>
          </div>

          {/* Action Buttons: Explain Prediction & Export */}
          <div className="pt-2 flex flex-wrap items-center justify-between gap-3 border-t border-slate-100">
            <div className="flex items-center space-x-3">
              <button
                onClick={() => {
                  const aid = (analysisResult as any)?.analysis_id;
                  navigate(aid ? `/explainability?analysis_id=${aid}` : '/explainability');
                }}
                className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-xs font-bold hover:bg-indigo-700 transition flex items-center shadow-xs"
              >
                <Eye className="w-3.5 h-3.5 mr-1.5" />
                View Full Explainability & Channel Attributions →
              </button>

              <button
                onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                className="px-4 py-2 bg-slate-100 text-slate-700 rounded-lg text-xs font-bold hover:bg-slate-200 transition"
              >
                {showTechnicalDetails ? 'Hide Technical Details' : 'View Technical Details'}
              </button>
            </div>

            <button
              onClick={handleExportReport}
              className="px-4 py-2 bg-slate-50 border border-slate-200 text-slate-700 rounded-lg text-xs font-bold hover:bg-slate-100 transition flex items-center"
            >
              <Download className="w-3.5 h-3.5 mr-1.5" />
              Export Prediction Report (JSON)
            </button>
          </div>

          {/* Expandable Technical Details Panel */}
          {showTechnicalDetails && (
            <div className="p-4 bg-slate-900 text-slate-200 rounded-xl font-mono text-xs space-y-2">
              <div className="text-slate-400 font-bold border-b border-slate-800 pb-1">
                // PyTorch Inference Telemetry & Tensor Dimensions
              </div>
              <div>Model: PyTorch EEGCNNTransformer (d_model=64, nhead=4, layers=2)</div>
              <div>Input Window Dimension: [16, 500] (16 Channels x 500 Samples at 250 Hz)</div>
              <div>Attentive Pooling: Window-level representation softmax aggregation</div>
              <div>Threshold: {uncertaintyThreshold} • Evidence Margin: {analysisResult.evidence_margin?.toFixed(4)}</div>
              <div>Checkpoint: backend/checkpoints/best_model.pt</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AnalyzeEEG;
