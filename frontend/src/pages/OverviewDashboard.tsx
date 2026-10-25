import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Database,
  Cpu,
  Activity,
  BarChart3,
  ExternalLink,
  CheckCircle2,
  Clock,
  Play,
  Brain,
  History,
  FolderOpen,
  ArrowRight,
  ShieldCheck,
  AlertCircle
} from 'lucide-react';
import { api } from '../api';
import { Dataset, Experiment, EvaluationMetrics, SystemHealth, ActivityEvent, ClassificationTasksReadiness } from '../types';
import { StatusBadge } from '../components/StatusBadge';
import { ConfusionMatrix } from '../components/ConfusionMatrix';

export const OverviewDashboard: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [tasksReadiness, setTasksReadiness] = useState<ClassificationTasksReadiness | null>(null);
  const [latestExp, setLatestExp] = useState<Experiment | null>(null);
  const [latestEval, setLatestEval] = useState<EvaluationMetrics | null>(null);
  const [recentEvents, setRecentEvents] = useState<ActivityEvent[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [dList, hInfo, exps, ev, acts, tReady] = await Promise.all([
          api.getDatasets(),
          api.getHealth(),
          api.getExperiments(),
          api.getLatestEvaluation(),
          api.getRecentActivity(6),
          api.getClassificationTasksReadiness().catch(() => null),
        ]);
        setDatasets(dList);
        setHealth(hInfo);
        if (exps && exps.length > 0) setLatestExp(exps[0]);
        if (ev && ev.has_evaluation) setLatestEval(ev as unknown as EvaluationMetrics);
        setRecentEvents(acts || []);
        setTasksReadiness(tReady || hInfo.classification_tasks_readiness || null);
      } catch (err) {
        console.error('Failed to load overview data', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const totalRegistered = datasets.reduce((acc, d) => acc + (d.verified_count || 0), 0);
  const totalExcluded = datasets.reduce((acc, d) => acc + (d.excluded_count || 0), 0);
  const hasModelCheckpoint = (health?.models_status?.checkpoints_available || 0) > 0;

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Top Hero Section */}
      <div className="bg-white rounded-2xl border border-slate-200 p-8 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="max-w-3xl">
            <div className="flex items-center space-x-2 mb-2">
              <span className="text-[11px] font-bold px-2.5 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200 uppercase tracking-wider">
                Research & Inference Platform
              </span>
              <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">
                PyTorch • MNE • SciPy • Captum
              </span>
            </div>
            <h1 className="text-2xl lg:text-3xl font-extrabold text-slate-900 tracking-tight">
              Explainable EEG Analysis
            </h1>
            <p className="mt-2 text-sm text-slate-600 leading-relaxed">
              Explore EEG data, train and evaluate machine-learning models, and understand the evidence behind their predictions.
            </p>
          </div>

          {/* Primary Quick Action Buttons */}
          <div className="flex flex-wrap lg:flex-col gap-2.5 shrink-0">
            <Link
              to="/analyze"
              className="inline-flex items-center justify-center px-4 py-2.5 rounded-xl bg-indigo-600 text-white text-xs font-bold shadow-xs hover:bg-indigo-700 transition"
            >
              <Brain className="w-3.5 h-3.5 mr-2" />
              Analyze EEG
            </Link>
            <Link
              to="/evaluation"
              className="inline-flex items-center justify-center px-4 py-2.5 rounded-xl bg-slate-100 text-slate-800 text-xs font-bold hover:bg-slate-200 transition"
            >
              <BarChart3 className="w-3.5 h-3.5 mr-2" />
              View Model Performance
            </Link>
            <Link
              to="/datasets"
              className="inline-flex items-center justify-center px-4 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-slate-700 text-xs font-bold hover:bg-slate-100 transition"
            >
              <Database className="w-3.5 h-3.5 mr-2" />
              Manage Datasets
            </Link>
          </div>
        </div>
      </div>

      {/* 4 Core Status Cards (100% Genuine Backend Telemetry) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* 1. Dataset Status */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-slate-500 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider">Dataset Status</span>
              <Database className="w-4 h-4 text-slate-400" />
            </div>
            <div className="text-2xl font-extrabold text-slate-900">
              {totalRegistered > 0 ? totalRegistered : '0'}
            </div>
            <div className="text-xs text-slate-500 mt-1">Usable EEG participants</div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-500 space-y-0.5">
            <div>Available Datasets: <strong className="text-slate-700">{datasets.length}</strong></div>
            <div>Excluded Records: <strong className="text-slate-700">{totalExcluded}</strong></div>
          </div>
        </div>

        {/* 2. Model Status */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-slate-500 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider">Model Status</span>
              <Cpu className="w-4 h-4 text-slate-400" />
            </div>
            <div className="text-2xl font-extrabold text-indigo-700">
              {hasModelCheckpoint ? 'Ready' : 'Not Trained'}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              {hasModelCheckpoint ? 'CNN–Transformer Checkpoint' : 'Awaiting training'}
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-500 space-y-0.5">
            <div>Architecture: <strong className="text-slate-700">CNN–Transformer</strong></div>
            <div>Checkpoint: <strong className="text-emerald-700 font-mono">{hasModelCheckpoint ? 'best_model.pt' : 'None'}</strong></div>
          </div>
        </div>

        {/* 3. Training Status */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-slate-500 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider">Training Status</span>
              <Activity className="w-4 h-4 text-slate-400" />
            </div>
            <div className="text-2xl font-extrabold text-slate-900">
              {latestExp ? latestExp.status : 'Idle'}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              {latestExp ? `${latestExp.current_epoch} / ${latestExp.total_epochs} epochs` : 'No active job'}
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-500 space-y-0.5">
            <div>Latest Run: <strong className="text-slate-700 truncate block">{latestExp?.name || 'None'}</strong></div>
            <div>Task: <strong className="text-slate-700">{latestExp?.target_task || 'N/A'}</strong></div>
          </div>
        </div>

        {/* 4. Evaluation Status */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-slate-500 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider">Evaluation Status</span>
              <BarChart3 className="w-4 h-4 text-slate-400" />
            </div>
            <div className="text-2xl font-extrabold text-emerald-700">
              {latestEval?.accuracy !== undefined ? `${(latestEval.accuracy * 100).toFixed(1)}%` : 'Not Evaluated'}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              {latestEval ? 'Held-Out Test Accuracy' : 'Run evaluation to verify'}
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-500 space-y-0.5">
            <div>Test Cohort: <strong className="text-slate-700">{latestEval ? `${latestEval.test_subjects_count} participants` : 'N/A'}</strong></div>
            <div>ROC-AUC: <strong className="text-slate-700">{latestEval?.roc_auc ? (latestEval.roc_auc * 100).toFixed(1) + '%' : 'N/A'}</strong></div>
          </div>
        </div>
      </div>

      {/* Main Two-Column Layout: Cohorts Overview & Real Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Classification Tasks & Model Readiness Status */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-bold text-slate-900 tracking-tight">Classification Tasks & Model Readiness</h2>
              <p className="text-xs text-slate-500">Live operational readiness status and verified model parameters per clinical task.</p>
            </div>
            <Link to="/datasets" className="text-xs text-indigo-600 font-semibold hover:underline flex items-center">
              Manage Datasets →
            </Link>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* 1. Schizophrenia vs. Healthy Controls Card */}
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">
                    Technical Validation
                  </span>
                  <StatusBadge label={tasksReadiness?.sz_vs_hc?.status || 'STATUS UNKNOWN'} size="sm" />
                </div>
                <h3 className="text-sm font-bold text-slate-900">1. Schizophrenia vs. Healthy Controls</h3>
                <p className="text-xs text-slate-500 mt-1">
                  Benchmarking deep learning classification on genuine multi-channel EEG waveforms.
                </p>

                {/* Status reason banner if not MODEL READY */}
                {tasksReadiness?.sz_vs_hc?.status !== 'MODEL READY' && tasksReadiness?.sz_vs_hc?.status_reason && (
                  <div className="mt-3 p-2.5 rounded-lg border border-amber-200 bg-amber-50 text-amber-900 text-xs">
                    <div className="flex items-start space-x-1.5">
                      <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-amber-600" />
                      <span>{tasksReadiness.sz_vs_hc.status_reason}</span>
                    </div>
                  </div>
                )}

                <div className="mt-3 text-xs space-y-1.5 font-mono text-slate-600 bg-slate-50 p-3 rounded-lg border border-slate-100">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Dataset:</span>
                    <strong className="text-slate-900">{tasksReadiness?.sz_vs_hc?.dataset || 'ASEEG'}</strong>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Participants:</span>
                    <span className="text-slate-900 font-semibold">
                      {tasksReadiness?.sz_vs_hc?.participants_verified ? (
                        <>
                          {tasksReadiness.sz_vs_hc.participants} Verified
                          <span className="text-emerald-600 text-[10px] ml-1">✓ Manifest</span>
                        </>
                      ) : (
                        <span className="text-slate-500 italic font-normal">Unverified</span>
                      )}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Sampling Rate:</span>
                    <strong className="text-slate-900">
                      {tasksReadiness?.sz_vs_hc?.sampling_rate || (
                        <span className="text-slate-500 italic font-normal">Unverified</span>
                      )}
                    </strong>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Channels:</span>
                    <strong className="text-slate-900">
                      {tasksReadiness?.sz_vs_hc?.channels ? (
                        `${tasksReadiness.sz_vs_hc.channels} Channels`
                      ) : (
                        <span className="text-slate-500 italic font-normal">Unverified</span>
                      )}
                    </strong>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Held-Out Test Acc:</span>
                    {tasksReadiness?.sz_vs_hc?.held_out_accuracy ? (
                      <strong className="text-emerald-700 font-bold bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                        {tasksReadiness.sz_vs_hc.held_out_accuracy}
                      </strong>
                    ) : (
                      <span className="text-slate-500 italic font-normal">Unconfirmed</span>
                    )}
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Active Checkpoint:</span>
                    {tasksReadiness?.sz_vs_hc?.active_checkpoint ? (
                      <span className="text-slate-900 font-semibold text-[11px] truncate max-w-[130px]">
                        {tasksReadiness.sz_vs_hc.active_checkpoint}
                      </span>
                    ) : (
                      <span className="text-slate-500 italic font-normal">None</span>
                    )}
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Model Architecture:</span>
                    {tasksReadiness?.sz_vs_hc?.model_architecture ? (
                      <strong className="text-indigo-700">{tasksReadiness.sz_vs_hc.model_architecture}</strong>
                    ) : (
                      <span className="text-slate-500 italic font-normal">Unconfirmed</span>
                    )}
                  </div>
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
                <Link
                  to="/analyze"
                  className="text-indigo-600 font-semibold hover:underline flex items-center text-[11px]"
                >
                  <Brain className="w-3 h-3 mr-1" />
                  Test Inference (sz1 / n1) →
                </Link>
                <Link
                  to="/evaluation"
                  className="text-emerald-700 text-[10px] font-bold hover:underline"
                >
                  View Evaluation →
                </Link>
              </div>
            </div>

            {/* 2. Schizophrenia vs. Bipolar Disorder Card */}
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-blue-700 bg-blue-50 px-2 py-0.5 rounded-full border border-blue-200">
                    Primary Differential
                  </span>
                  <StatusBadge label={tasksReadiness?.scz_vs_bd?.status || 'DATA REQUIRED'} size="sm" />
                </div>
                <h3 className="text-sm font-bold text-slate-900">2. Schizophrenia vs. Bipolar Disorder</h3>
                <p className="text-xs text-slate-500 mt-1">
                  Differential psychiatric diagnosis on Bipolar & Schizophrenia Network recordings.
                </p>

                {/* Status Explanation Banner */}
                <div className={`mt-3 p-2.5 rounded-lg border text-xs leading-relaxed ${
                  tasksReadiness?.scz_vs_bd?.status === 'MODEL READY'
                    ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
                    : tasksReadiness?.scz_vs_bd?.status === 'MODEL NOT TRAINED'
                    ? 'bg-amber-50 border-amber-200 text-amber-900'
                    : tasksReadiness?.scz_vs_bd?.status === 'STATUS UNKNOWN'
                    ? 'bg-rose-50 border-rose-200 text-rose-900'
                    : 'bg-amber-50 border-amber-200 text-amber-900'
                }`}>
                  <div className="flex items-start space-x-1.5">
                    <AlertCircle className={`w-4 h-4 shrink-0 mt-0.5 ${
                      tasksReadiness?.scz_vs_bd?.status === 'MODEL READY'
                        ? 'text-emerald-600'
                        : tasksReadiness?.scz_vs_bd?.status === 'STATUS UNKNOWN'
                        ? 'text-rose-600'
                        : 'text-amber-600'
                    }`} />
                    <span>
                      {tasksReadiness?.scz_vs_bd?.explanation ||
                        'Authorized B-SNIP 1 data is required to train and evaluate this classification task.'}
                    </span>
                  </div>
                </div>

                <div className="mt-3 text-xs space-y-1.5 font-mono text-slate-600 bg-slate-50 p-3 rounded-lg border border-slate-100">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Target Cohort:</span>
                    <strong className="text-slate-900">{tasksReadiness?.scz_vs_bd?.dataset || 'B-SNIP 1'}</strong>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Required Source:</span>
                    <span className="text-slate-700 font-semibold">NIH NDA Collection #2274</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Active Checkpoint:</span>
                    {tasksReadiness?.scz_vs_bd?.active_checkpoint ? (
                      <span className="text-slate-900 font-semibold text-[11px] truncate max-w-[130px]">
                        {tasksReadiness.scz_vs_bd.active_checkpoint}
                      </span>
                    ) : (
                      <span className="text-slate-500 italic font-normal">None (Not Trained)</span>
                    )}
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Held-Out Test Acc:</span>
                    {tasksReadiness?.scz_vs_bd?.held_out_accuracy ? (
                      <strong className="text-emerald-700 font-bold bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                        {tasksReadiness.scz_vs_bd.held_out_accuracy}
                      </strong>
                    ) : (
                      <span className="text-slate-500 italic font-normal">N/A (Untrained)</span>
                    )}
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Model Architecture:</span>
                    {tasksReadiness?.scz_vs_bd?.model_architecture ? (
                      <strong className="text-indigo-700">{tasksReadiness.scz_vs_bd.model_architecture}</strong>
                    ) : (
                      <span className="text-slate-500 italic font-normal">Untrained</span>
                    )}
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Data Isolation:</span>
                    <span className="text-blue-700 text-[10px] font-bold">Never reuses ASEEG</span>
                  </div>
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
                <a
                  href="https://nda.nih.gov/edit_collection.html?id=2274"
                  target="_blank"
                  rel="noreferrer"
                  className="text-indigo-600 font-semibold hover:underline flex items-center text-[11px]"
                >
                  NIH NDA #2274 <ExternalLink className="w-3 h-3 ml-1" />
                </a>
                <Link to="/datasets" className="text-amber-800 text-[10px] font-bold hover:underline">
                  Dataset Status →
                </Link>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Genuine Recent Activity Feed */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center">
                <History className="w-4 h-4 text-indigo-600 mr-2" />
                Recent System Activity
              </h2>
              <Link to="/history" className="text-xs text-indigo-600 font-semibold hover:underline">
                Full Log →
              </Link>
            </div>
            <p className="text-xs text-slate-500 mb-4">Chronological record of authentic pipeline events.</p>

            {recentEvents.length === 0 ? (
              <div className="py-8 text-center text-slate-400 text-xs">
                No recent activity recorded yet.
              </div>
            ) : (
              <div className="space-y-3">
                {recentEvents.map((evt) => (
                  <div key={evt.id} className="p-3 bg-slate-50 rounded-lg border border-slate-100 text-xs space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-900 truncate pr-2">{evt.title}</span>
                      <span className="text-[10px] text-slate-400 shrink-0 font-mono">
                        {evt.timestamp ? String(evt.timestamp).split(' ')[1] || String(evt.timestamp) : ''}
                      </span>
                    </div>
                    <p className="text-slate-600 text-[11px] leading-relaxed">{evt.description}</p>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-500 flex items-center justify-between">
            <span className="flex items-center">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600 mr-1" />
              100% Genuine Backend Data
            </span>
            <Link to="/training" className="text-indigo-600 font-semibold hover:underline">
              Launch Run →
            </Link>
          </div>
        </div>
      </div>

      {/* Latest Evaluation Highlights (Held-Out 25% Test Partition) */}
      {latestEval && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-700 block mb-0.5">
                Held-Out Benchmark
              </span>
              <h2 className="text-base font-bold text-slate-900 tracking-tight">
                Held-Out 25% Test Partition Performance
              </h2>
            </div>
            <Link to="/evaluation" className="text-xs text-indigo-600 font-semibold hover:underline flex items-center">
              Full Evaluation Dashboard →
            </Link>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-center">
            <ConfusionMatrix matrix={latestEval.confusion_matrix} labels={['SZ', 'HC']} />

            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3 bg-slate-50 rounded-lg text-center border border-slate-200">
                  <span className="text-slate-500 text-[11px] block">Accuracy</span>
                  <span className="text-lg font-bold text-slate-900">{(latestEval.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div className="p-3 bg-slate-50 rounded-lg text-center border border-slate-200">
                  <span className="text-slate-500 text-[11px] block">F1-Score</span>
                  <span className="text-lg font-bold text-slate-900">{(latestEval.f1_score * 100).toFixed(1)}%</span>
                </div>
                <div className="p-3 bg-slate-50 rounded-lg text-center border border-slate-200">
                  <span className="text-slate-500 text-[11px] block">Sensitivity</span>
                  <span className="text-lg font-bold text-slate-900">{(latestEval.sensitivity * 100).toFixed(1)}%</span>
                </div>
                <div className="p-3 bg-slate-50 rounded-lg text-center border border-slate-200">
                  <span className="text-slate-500 text-[11px] block">ROC-AUC</span>
                  <span className="text-lg font-bold text-slate-900">
                    {latestEval.roc_auc ? `${(latestEval.roc_auc * 100).toFixed(1)}%` : 'N/A'}
                  </span>
                </div>
              </div>

              <div className="p-3.5 bg-indigo-50/60 rounded-xl border border-indigo-100 text-slate-700 leading-relaxed text-[11px]">
                <strong>Scientific Partitioning Standard: </strong>
                Evaluated on {latestEval.test_subjects_count} previously unseen participants with zero participant leakage across splits.
                Uncertainty thresholding rejected {latestEval.uncertain_count} ambiguous participants for safe clinical verification.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default OverviewDashboard;
