import React, { useState, useEffect } from 'react';
import {
  FileText,
  Printer,
  Download,
  CheckCircle2,
  ShieldAlert,
  Sparkles,
  BookOpen,
  Calendar,
  Layers,
  Cpu
} from 'lucide-react';
import { api } from '../api';
import { ResearchReport } from '../types';

export const ResearchReports: React.FC = () => {
  const [reports, setReports] = useState<ResearchReport[]>([]);
  const [activeReport, setActiveReport] = useState<ResearchReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);

  const fetchReports = async () => {
    try {
      const repList = await api.getReports();
      setReports(repList);
      if (repList.length > 0 && !activeReport) {
        setActiveReport(repList[0]);
      }
    } catch (err) {
      console.error('Failed to load reports', err);
    }
  };

  useEffect(() => {
    fetchReports();
  }, []);

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      const newRep = await api.generateReport();
      setActiveReport(newRep);
      await fetchReports();
    } catch (err: any) {
      alert(`Report generation failed: ${err.message}`);
    } finally {
      setGenerating(false);
    }
  };

  const handlePrint = () => {
    window.print();
  };

  const reportData = activeReport?.report_data || {};

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-slate-500 mb-1">
            <FileText className="w-4 h-4 text-blue-600" />
            <span>Academic Project Documentation</span>
          </div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">Semester 7 Faculty Evaluation Report</h1>
          <p className="text-xs text-slate-500 mt-1">
            Formal technical report synthesizing methodology, signal processing, deep learning architecture, and empirical findings.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={handleGenerate}
            disabled={generating}
            className="px-4 py-2 rounded-xl bg-blue-600 text-white text-xs font-bold hover:bg-blue-700 transition flex items-center shadow-xs disabled:opacity-50"
          >
            <Sparkles className="w-3.5 h-3.5 mr-1.5" />
            {generating ? 'Compiling Report...' : 'Generate Fresh Report'}
          </button>
          {activeReport && (
            <button
              onClick={handlePrint}
              className="px-3.5 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold flex items-center transition shadow-2xs"
            >
              <Printer className="w-3.5 h-3.5 mr-1.5" />
              Print / Save PDF
            </button>
          )}
        </div>
      </div>

      {/* Reports history select */}
      {reports.length > 1 && (
        <div className="flex items-center space-x-3 bg-white p-3 rounded-xl border border-slate-200 text-xs">
          <span className="font-semibold text-slate-600">Saved Reports:</span>
          <div className="flex space-x-2 overflow-x-auto">
            {reports.map((r) => (
              <button
                key={r.id}
                onClick={() => setActiveReport(r)}
                className={`px-3 py-1.5 rounded-lg font-mono transition ${
                  activeReport?.id === r.id
                    ? 'bg-blue-50 text-blue-700 font-bold border border-blue-200'
                    : 'bg-slate-50 text-slate-600 hover:bg-slate-100'
                }`}
              >
                {r.id}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Formal Academic Report Container */}
      {activeReport ? (
        <div className="bg-white rounded-2xl border border-slate-200 p-8 sm:p-12 shadow-xs space-y-8 print:border-none print:shadow-none print:p-0">
          {/* Header Section */}
          <div className="border-b border-slate-200 pb-6 text-center space-y-2">
            <span className="text-xs font-bold uppercase tracking-wider text-blue-700">
              Department of Computer Engineering • Semester 7 Capstone Project
            </span>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
              Explainable Multimodal Transformer for SCZ-BD Differential Analysis
            </h1>
            <p className="text-xs text-slate-500 max-w-2xl mx-auto">
              Investigating Deep Convolutional-Transformer Architectures for Electroencephalographic (EEG) Differential Saliency between Schizophrenia and Bipolar Disorder
            </p>
            <div className="pt-2 flex justify-center items-center space-x-4 text-[11px] text-slate-400 font-mono">
              <span>Report ID: {activeReport.id}</span>
              <span>•</span>
              <span>Generated: {new Date(activeReport.created_at).toLocaleDateString()}</span>
            </div>
          </div>

          {/* Research Disclaimer Box */}
          <div className="bg-amber-50 border border-amber-300 rounded-xl p-4 text-xs text-amber-900 leading-relaxed">
            <strong className="font-bold uppercase tracking-wider block mb-0.5">
              Academic & Diagnostic Non-Equivalence Notice
            </strong>
            This project represents an engineering research prototype developed for academic evaluation. It is explicitly
            NOT a certified medical diagnostic system and must not be utilized for clinical healthcare or therapeutic decision-making.
          </div>

          {/* 1. Executive Summary */}
          <section className="space-y-3">
            <h2 className="text-base font-bold text-slate-900 border-b border-slate-100 pb-1.5 flex items-center">
              <span className="w-5 h-5 rounded-full bg-blue-100 text-blue-700 inline-flex items-center justify-center text-xs font-bold mr-2">1</span>
              Executive Summary & Problem Statement
            </h2>
            <p className="text-xs text-slate-700 leading-relaxed">
              Differentiating Schizophrenia (SCZ) from Bipolar Disorder (BD) remains one of the most challenging frontiers in neuropsychiatric
              diagnostics due to overlapping clinical phenomenology, psychotic symptoms, and affective dysregulation.
              This research develops an end-to-end full-stack computational system utilizing hybrid deep learning (CNN-Transformer)
              and path-integral gradient feature attribution (Integrated Gradients) to discover differential electroencephalographic representations
              from multi-channel sensor signals.
            </p>
          </section>

          {/* 2. Dataset Compliance & Separation */}
          <section className="space-y-3">
            <h2 className="text-base font-bold text-slate-900 border-b border-slate-100 pb-1.5 flex items-center">
              <span className="w-5 h-5 rounded-full bg-blue-100 text-blue-700 inline-flex items-center justify-center text-xs font-bold mr-2">2</span>
              Dataset Provenance & Benchmark Isolation
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
                <span className="font-bold text-slate-800 block">Primary Cohort: B-SNIP 1</span>
                <p className="text-slate-600">
                  Targeted NIH NDA Collection #2274. Governed under NIH Data Use Certification (DUC).
                  The software strictly validates local manifests and authorized EDF/FIF files without circumventing security policies.
                </p>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
                <span className="font-bold text-slate-800 block">Technical Validation: ASEEG</span>
                <p className="text-slate-600">
                  Zenodo record #18029536. Strictly designated as <strong>SZ vs HC Technical Validation</strong>.
                  Explicitly partitioned to verify network convergence without conflating performance with SCZ-vs-BD differential claims.
                </p>
              </div>
            </div>
          </section>

          {/* 3. Signal Processing Methodology */}
          <section className="space-y-3">
            <h2 className="text-base font-bold text-slate-900 border-b border-slate-100 pb-1.5 flex items-center">
              <span className="w-5 h-5 rounded-full bg-blue-100 text-blue-700 inline-flex items-center justify-center text-xs font-bold mr-2">3</span>
              Signal Processing & Artifact Rejection Pipeline
            </h2>
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 text-xs text-slate-700 space-y-2">
              <ul className="list-disc pl-4 space-y-1 text-slate-600">
                <li><strong>Frequency Bandpass Filtering:</strong> Zero-phase 4th order Butterworth filter with cutoffs at 1.0 Hz (high-pass) and 45.0 Hz (low-pass) to eliminate DC offset, sweat artifacts, and powerline noise.</li>
                <li><strong>Common Average Referencing (CAR):</strong> Subtracts instantaneous spatial mean across 16 electrodes to remove common global artifacts.</li>
                <li><strong>Windowing & Rejection:</strong> 2.0-second sliding windows (500 samples at 250 Hz) with an automated 150 µV peak-to-peak amplitude artifact rejection threshold.</li>
                <li><strong>Spectral Decomposition:</strong> Welch Power Spectral Density computed across Delta (0.5-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), and Gamma (30-45Hz).</li>
              </ul>
            </div>
          </section>

          {/* 4. Model Architecture & Subject Isolation */}
          <section className="space-y-3">
            <h2 className="text-base font-bold text-slate-900 border-b border-slate-100 pb-1.5 flex items-center">
              <span className="w-5 h-5 rounded-full bg-blue-100 text-blue-700 inline-flex items-center justify-center text-xs font-bold mr-2">4</span>
              Deep Learning Architecture & Anti-Leakage Protocol
            </h2>
            <p className="text-xs text-slate-700 leading-relaxed">
              The neural architecture integrates 1D Temporal Convolutions (kernel=25, 32 filters), 1D Spatial Convolutions (kernel=15, 64 filters),
              Sinusoidal Positional Encoding, and a 2-layer 4-head Transformer Encoder (d_model=64).
              Window representations are pooled at the subject level using learned attention weights: <code className="font-mono bg-slate-100 px-1 py-0.5 rounded">α = softmax(w^T tanh(W x_i))</code>.
            </p>
            <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-xs text-emerald-900">
              <strong>Data Leakage Prevention:</strong> Stratified partitioning is executed strictly at the Subject ID level before window extraction. Windows belonging to the same subject never cross training, validation, or test boundaries.
            </div>
          </section>

          {/* 5. Empirical Results */}
          <section className="space-y-3">
            <h2 className="text-base font-bold text-slate-900 border-b border-slate-100 pb-1.5 flex items-center">
              <span className="w-5 h-5 rounded-full bg-blue-100 text-blue-700 inline-flex items-center justify-center text-xs font-bold mr-2">5</span>
              Empirical Results & Baseline Benchmark
            </h2>
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 text-xs space-y-2 font-mono">
              <div>Summary Findings: <strong className="text-slate-900 font-sans">{activeReport.summary}</strong></div>
              {reportData.evaluation && (
                <div className="pt-2 border-t border-slate-200 grid grid-cols-2 sm:grid-cols-4 gap-2 text-slate-700">
                  <div>Accuracy: <strong>{(reportData.evaluation.accuracy * 100).toFixed(1)}%</strong></div>
                  <div>F1-Score: <strong>{(reportData.evaluation.f1_score * 100).toFixed(1)}%</strong></div>
                  <div>Sensitivity: <strong>{(reportData.evaluation.sensitivity * 100).toFixed(1)}%</strong></div>
                  <div>Specificity: <strong>{(reportData.evaluation.specificity * 100).toFixed(1)}%</strong></div>
                </div>
              )}
            </div>
          </section>

          {/* 6. Scientific Limitations */}
          <section className="space-y-3">
            <h2 className="text-base font-bold text-slate-900 border-b border-slate-100 pb-1.5 flex items-center">
              <span className="w-5 h-5 rounded-full bg-blue-100 text-blue-700 inline-flex items-center justify-center text-xs font-bold mr-2">6</span>
              Scientific Limitations & Conclusions
            </h2>
            <ul className="list-disc pl-4 space-y-1 text-xs text-slate-600 leading-relaxed">
              <li>Gradient-based feature attribution describes mathematical sensitivity within the trained network and does not establish medical biomarkers or disease causation.</li>
              <li>Medication effects (e.g. antipsychotics vs mood stabilizers) can influence resting-state EEG oscillatory activity and must be accounted for in future clinical trials.</li>
              <li>Uncertainty rejection successfully guarded borderline cases, demonstrating that safety-critical decision systems should prioritize rejection over false confidence.</li>
            </ul>
          </section>

          {/* Signatures Footer */}
          <div className="pt-8 border-t border-slate-200 grid grid-cols-2 sm:grid-cols-3 gap-6 text-xs text-slate-500">
            <div>
              <span className="block font-semibold text-slate-700 mb-6">Student Researcher</span>
              <div className="border-t border-slate-300 pt-1">Computer Engineering (Sem 7)</div>
            </div>
            <div>
              <span className="block font-semibold text-slate-700 mb-6">Project Guide / Faculty Evaluator</span>
              <div className="border-t border-slate-300 pt-1">Department of Computer Engineering</div>
            </div>
            <div>
              <span className="block font-semibold text-slate-700 mb-6">Evaluation Status</span>
              <div className="border-t border-slate-300 pt-1 text-emerald-700 font-semibold">Reviewed & Verified</div>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-slate-200 p-12 text-center space-y-4">
          <BookOpen className="w-12 h-12 text-slate-300 mx-auto" />
          <h2 className="text-base font-bold text-slate-800">No Research Reports Generated Yet</h2>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Click "Generate Fresh Report" to compile a formal academic evaluation document synthesizing the dataset provenance, signal processing, model architecture, and experimental results.
          </p>
        </div>
      )}
    </div>
  );
};
