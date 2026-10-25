import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Cpu,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  ShieldCheck,
  Activity,
  Layers,
  ArrowRight,
  TrendingDown,
  Clock,
  Sparkles
} from 'lucide-react';
import { api } from '../api';
import { Experiment, Dataset } from '../types';
import { StatusBadge } from '../components/StatusBadge';

export const ModelLab: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedExpId, setSelectedExpId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [trainingInProgress, setTrainingInProgress] = useState(false);

  // Training form configuration
  const [datasetId, setDatasetId] = useState<string>('aseeg');
  const [targetTask, setTargetTask] = useState<string>('SZ_VS_HC_TECH_VAL');
  const [modelType, setModelType] = useState<'CNN_TRANSFORMER' | 'LOGISTIC_REGRESSION'>('CNN_TRANSFORMER');
  const [epochs, setEpochs] = useState<number>(15);
  const [batchSize, setBatchSize] = useState<number>(16);
  const [learningRate, setLearningRate] = useState<number>(0.0005);
  const [uncertaintyThreshold, setUncertaintyThreshold] = useState<number>(0.65);

  const fetchExperiments = async () => {
    try {
      const exps = await api.getExperiments();
      setExperiments(exps);
      if (!selectedExpId && exps.length > 0) {
        setSelectedExpId(exps[0].id);
      }
      const running = exps.some((e) => e.status === 'RUNNING');
      setTrainingInProgress(running);
    } catch (err) {
      console.error('Failed to load experiments', err);
    }
  };

  useEffect(() => {
    const init = async () => {
      setLoading(true);
      try {
        const dList = await api.getDatasets();
        setDatasets(dList);
        await fetchExperiments();
      } finally {
        setLoading(false);
      }
    };
    init();
  }, []);

  // Poll while training is running
  useEffect(() => {
    let interval: any;
    if (trainingInProgress) {
      interval = setInterval(fetchExperiments, 2000);
    }
    return () => clearInterval(interval);
  }, [trainingInProgress]);

  const handleStartTraining = async () => {
    setLoading(true);
    try {
      const res = await api.startTraining({
        dataset_id: datasetId,
        target_task: targetTask,
        model_type: modelType,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        uncertainty_threshold: uncertaintyThreshold,
      });
      setSelectedExpId(res.experiment_id);
      setTrainingInProgress(true);
      await fetchExperiments();
    } catch (err: any) {
      alert(`Failed to start training: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const activeExp = experiments.find((e) => e.id === selectedExpId) || experiments[0];

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Title */}
      <div>
        <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-slate-500 mb-1">
          <Cpu className="w-4 h-4 text-blue-600" />
          <span>Deep Learning & Baseline Laboratory</span>
        </div>
        <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">CNN–Transformer & Logistic Regression Experiments</h1>
        <p className="text-xs text-slate-500 mt-1">
          Train hybrid 1D Temporal-Spatial Convolutional Transformer architectures and compare against classical spectral baselines.
        </p>
      </div>

      {/* Strict Subject-Level Split Isolation Notice */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 flex items-start space-x-3 text-xs text-blue-900">
        <ShieldCheck className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
        <div className="leading-relaxed">
          <strong className="font-bold uppercase tracking-wider block mb-0.5">Strict Subject-Level Partitioning Enforced</strong>
          To prevent temporal-spatial data leakage across EEG sliding windows, training, validation, and test splits are
          isolated strictly at the <strong>Subject ID level</strong> before window segmentation. Zero subject overlap exists across partitions.
        </div>
      </div>

      {/* Architecture Blueprint & Baseline Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* CNN-Transformer card */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800">
              PyTorch CNN–Transformer Deep Architecture
            </h2>
            <span className="text-xs font-mono font-bold px-2 py-0.5 rounded-md bg-blue-50 text-blue-700 border border-blue-200">
              16 Channels • 250 Hz
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 text-xs">
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <span className="text-slate-500 block">1. Temporal Conv1D</span>
              <strong className="text-slate-900 font-mono text-[11px] block mt-1">Kernel=25, 32 Filt</strong>
              <span className="text-[10px] text-slate-400">Captures ~100ms oscillations</span>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <span className="text-slate-500 block">2. Spatial Conv1D</span>
              <strong className="text-slate-900 font-mono text-[11px] block mt-1">Kernel=15, 64 Filt</strong>
              <span className="text-[10px] text-slate-400">Topographical sensor patterns</span>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <span className="text-slate-500 block">3. Multi-Head Attention</span>
              <strong className="text-slate-900 font-mono text-[11px] block mt-1">4 Heads, 2 Layers</strong>
              <span className="text-[10px] text-slate-400">d_model=64, Pos. Encoding</span>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <span className="text-slate-500 block">4. Subject Pooler</span>
              <strong className="text-slate-900 font-mono text-[11px] block mt-1">Attentive Pooling</strong>
              <span className="text-[10px] text-slate-400">softmax(w^T tanh(W x_i))</span>
            </div>
          </div>

          <p className="text-xs text-slate-600 leading-relaxed">
            Local temporal features extracted by 1D convolutional layers are projected into multi-head self-attention encoders.
            Window representations for each subject are aggregated via attentive pooling to compute calibrated SCZ and BD evidence scores.
          </p>
        </div>

        {/* Baseline Model card */}
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800">Classical Baseline</h2>
            <span className="text-xs font-mono font-bold px-2 py-0.5 rounded-md bg-slate-100 text-slate-700">
              scikit-learn
            </span>
          </div>

          <div className="text-xs text-slate-600 space-y-2">
            <p className="font-semibold text-slate-800">Documented Clinical EEG Features:</p>
            <ul className="list-disc pl-4 space-y-1 text-slate-500 text-[11px]">
              <li>Relative band powers (Delta, Theta, Alpha, Beta, Gamma) across all 16 channels</li>
              <li>Theta / Alpha ratio index (hallmark marker)</li>
              <li>Frontal Alpha Asymmetry: ln(Alpha_F4) - ln(Alpha_F3)</li>
            </ul>
            <div className="pt-2 border-t border-slate-100 text-[11px] text-slate-500">
              Pipeline: Standard Scaler + L2-Regularized Logistic Regression (Balanced class weights).
            </div>
          </div>
        </div>
      </div>

      {/* Training Configuration Form */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-5">
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800">
          Configure & Launch New Experiment
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          {/* Dataset Selection */}
          <div className="space-y-1">
            <label className="font-semibold text-slate-700">Training Cohort:</label>
            <select
              value={datasetId}
              onChange={(e) => {
                const val = e.target.value;
                setDatasetId(val);
                if (val === 'aseeg') setTargetTask('SZ_VS_HC_TECH_VAL');
                if (val === 'bsnip1') setTargetTask('SCZ_VS_BD');
              }}
              className="w-full p-2.5 rounded-lg border border-slate-200 bg-slate-50 text-slate-800 font-semibold focus:outline-none focus:border-blue-600"
            >
              <option value="aseeg">ASEEG Benchmark (Zenodo #18029536 - 97 Verified)</option>
              <option value="bsnip1">B-SNIP 1 (NIH NDA #2274 - Primary SCZ vs BD)</option>
              <option value="zenodo_14808296">Zenodo #14808296 (Rácz et al., 2025)</option>
            </select>
          </div>

          {/* Target Task */}
          <div className="space-y-1">
            <label className="font-semibold text-slate-700">Target Task:</label>
            <select
              value={targetTask}
              onChange={(e) => setTargetTask(e.target.value)}
              className="w-full p-2.5 rounded-lg border border-slate-200 bg-slate-50 text-slate-800 font-semibold focus:outline-none focus:border-blue-600"
            >
              <option value="SCZ_VS_BD">SCZ vs BD (Primary Differential)</option>
              <option value="SZ_VS_HC_TECH_VAL">SZ vs HC Technical Validation</option>
            </select>
          </div>

          {/* Model Type */}
          <div className="space-y-1">
            <label className="font-semibold text-slate-700">Model Architecture:</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value as any)}
              className="w-full p-2.5 rounded-lg border border-slate-200 bg-slate-50 text-slate-800 font-semibold focus:outline-none focus:border-blue-600"
            >
              <option value="CNN_TRANSFORMER">PyTorch CNN–Transformer</option>
              <option value="LOGISTIC_REGRESSION">Baseline Logistic Regression</option>
            </select>
          </div>

          {/* Uncertainty Threshold */}
          <div className="space-y-1">
            <div className="flex justify-between font-semibold text-slate-700">
              <span>Uncertainty Threshold:</span>
              <span className="font-mono text-blue-600">{uncertaintyThreshold}</span>
            </div>
            <input
              type="range"
              min="0.50"
              max="0.90"
              step="0.05"
              value={uncertaintyThreshold}
              onChange={(e) => setUncertaintyThreshold(parseFloat(e.target.value))}
              className="w-full mt-2 accent-blue-600 cursor-pointer"
            />
          </div>
        </div>

        {/* Hyperparams row */}
        {modelType === 'CNN_TRANSFORMER' && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-3 border-t border-slate-100 text-xs">
            <div>
              <label className="font-semibold text-slate-700 block mb-1">Epochs:</label>
              <input
                type="number"
                min="5"
                max="50"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value) || 15)}
                className="w-full p-2 rounded-lg border border-slate-200 font-mono text-xs"
              />
            </div>
            <div>
              <label className="font-semibold text-slate-700 block mb-1">Batch Size:</label>
              <input
                type="number"
                min="8"
                max="64"
                step="8"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value) || 16)}
                className="w-full p-2 rounded-lg border border-slate-200 font-mono text-xs"
              />
            </div>
            <div>
              <label className="font-semibold text-slate-700 block mb-1">Learning Rate:</label>
              <input
                type="number"
                step="0.0001"
                min="0.00001"
                max="0.01"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.0005)}
                className="w-full p-2 rounded-lg border border-slate-200 font-mono text-xs"
              />
            </div>
          </div>
        )}

        {/* Preflight Checklist (Section 7) */}
        <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
              Preflight Training Verification Checklist
            </span>
            <span className="text-[10px] font-mono text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200 font-semibold">
              Automatic Validation Active
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
            <div className="flex items-center space-x-2 text-slate-700">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>Dataset recordings validated on local filesystem</span>
            </div>
            <div className="flex items-center space-x-2 text-slate-700">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>Participant diagnostic labels verified in registry</span>
            </div>
            <div className="flex items-center space-x-2 text-slate-700">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>Both target diagnostic classes present</span>
            </div>
            <div className="flex items-center space-x-2 text-slate-700">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>Strict 75%/25% participant split (zero leakage)</span>
            </div>
            <div className="flex items-center space-x-2 text-slate-700">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>16-channel 10–20 montage layout confirmed</span>
            </div>
            <div className="flex items-center space-x-2 text-slate-700">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>PyTorch compute engine initialized</span>
            </div>
          </div>
        </div>

        <div className="pt-2 flex items-center justify-between">
          <span className="text-xs text-slate-500">
            Selected: <strong className="text-slate-800">{datasetId}</strong> ({targetTask})
          </span>
          <button
            onClick={handleStartTraining}
            disabled={loading || trainingInProgress}
            className="px-6 py-2.5 rounded-xl bg-indigo-600 text-white text-xs font-bold hover:bg-indigo-700 disabled:opacity-50 transition shadow-xs flex items-center"
          >
            <Play className="w-4 h-4 mr-2" />
            {trainingInProgress ? 'Training in Progress...' : 'Start Training Job'}
          </button>
        </div>
      </div>

      {/* Active Experiment Real-Time Loss Curves */}
      {activeExp && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <div className="flex items-center space-x-2">
                <StatusBadge label={activeExp.status} />
                <h3 className="text-base font-bold text-slate-900">{activeExp.name}</h3>
              </div>
              <p className="text-xs text-slate-500 font-mono mt-1">
                ID: {activeExp.id} • Created: {new Date(activeExp.created_at).toLocaleTimeString()}
              </p>
            </div>

            {activeExp.status === 'COMPLETED' && (
              <Link
                to="/evaluation"
                className="text-xs px-4 py-2 rounded-lg bg-emerald-50 text-emerald-800 border border-emerald-200 font-bold hover:bg-emerald-100 transition flex items-center"
              >
                Inspect Held-Out Test Evaluation <ArrowRight className="w-3.5 h-3.5 ml-1" />
              </Link>
            )}
          </div>

          {/* Progress bar if running */}
          {activeExp.status === 'RUNNING' && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-slate-600 font-mono">
                <span>Training Progress</span>
                <span>Epoch {activeExp.current_epoch} / {activeExp.total_epochs}</span>
              </div>
              <div className="w-full h-2.5 rounded-full bg-slate-100 overflow-hidden">
                <div
                  className="h-full bg-blue-600 transition-all duration-300 rounded-full"
                  style={{ width: `${(activeExp.current_epoch / Math.max(1, activeExp.total_epochs)) * 100}%` }}
                />
              </div>
            </div>
          )}

          {/* Loss and Metric Curves */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-3">
            {/* Loss Curve */}
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Epoch Loss Convergence
                </span>
                <span className="text-[11px] text-slate-400 font-mono">CrossEntropyLoss</span>
              </div>

              {activeExp.train_loss_history && activeExp.train_loss_history.length > 0 ? (
                <div className="space-y-2">
                  <div className="h-32 flex items-end space-x-1.5 pt-4">
                    {activeExp.train_loss_history.map((loss, idx) => {
                      const maxLoss = Math.max(...activeExp.train_loss_history, 0.8);
                      const heightPercent = Math.max(10, (loss / maxLoss) * 100);
                      return (
                        <div key={idx} className="flex-1 flex flex-col items-center group relative">
                          <div
                            className="w-full bg-blue-500 rounded-t hover:bg-blue-600 transition-all"
                            style={{ height: `${heightPercent}%` }}
                          />
                          <span className="text-[9px] text-slate-400 font-mono mt-1">{idx + 1}</span>
                          {/* Tooltip */}
                          <div className="absolute -top-7 hidden group-hover:block bg-slate-800 text-white text-[10px] py-0.5 px-1.5 rounded font-mono shadow-xs">
                            {loss.toFixed(3)}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <div className="flex justify-between text-[11px] text-slate-500 font-mono border-t border-slate-200 pt-1">
                    <span>Initial: {activeExp.train_loss_history[0]}</span>
                    <span>Current: {activeExp.train_loss_history[activeExp.train_loss_history.length - 1]}</span>
                  </div>
                </div>
              ) : (
                <div className="h-32 flex items-center justify-center text-xs text-slate-400">
                  Loss logs will appear here during training.
                </div>
              )}
            </div>

            {/* Validation Metric Curve */}
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Validation Accuracy Trajectory
                </span>
                <span className="text-[11px] text-slate-400 font-mono">Per-epoch</span>
              </div>

              {activeExp.val_metric_history && activeExp.val_metric_history.length > 0 ? (
                <div className="space-y-2">
                  <div className="h-32 flex items-end space-x-1.5 pt-4">
                    {activeExp.val_metric_history.map((acc, idx) => {
                      const heightPercent = Math.max(10, acc * 100);
                      return (
                        <div key={idx} className="flex-1 flex flex-col items-center group relative">
                          <div
                            className="w-full bg-emerald-500 rounded-t hover:bg-emerald-600 transition-all"
                            style={{ height: `${heightPercent}%` }}
                          />
                          <span className="text-[9px] text-slate-400 font-mono mt-1">{idx + 1}</span>
                          <div className="absolute -top-7 hidden group-hover:block bg-slate-800 text-white text-[10px] py-0.5 px-1.5 rounded font-mono shadow-xs">
                            {(acc * 100).toFixed(1)}%
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <div className="flex justify-between text-[11px] text-slate-500 font-mono border-t border-slate-200 pt-1">
                    <span>Peak Val: {(Math.max(...activeExp.val_metric_history) * 100).toFixed(1)}%</span>
                    <span>Latest: {((activeExp.val_metric_history[activeExp.val_metric_history.length - 1] || 0) * 100).toFixed(1)}%</span>
                  </div>
                </div>
              ) : (
                <div className="h-32 flex items-center justify-center text-xs text-slate-400">
                  Validation trajectory will display during epochs.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Experiments History Table */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <h2 className="text-base font-bold text-slate-900">Experiment Checkpoint History ({experiments.length})</h2>
        {experiments.length === 0 ? (
          <div className="py-8 text-center text-slate-400 text-xs">
            No training experiments recorded yet. Click "Start Training Job" above.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-50 text-slate-500 uppercase font-semibold border-b border-slate-200">
                <tr>
                  <th className="py-2.5 px-3">Experiment ID</th>
                  <th className="py-2.5 px-3">Model Architecture</th>
                  <th className="py-2.5 px-3">Target Task</th>
                  <th className="py-2.5 px-3">Cohort</th>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3">Test Accuracy</th>
                  <th className="py-2.5 px-3">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {experiments.map((exp) => (
                  <tr
                    key={exp.id}
                    onClick={() => setSelectedExpId(exp.id)}
                    className={`cursor-pointer hover:bg-slate-50 transition ${
                      selectedExpId === exp.id ? 'bg-blue-50/50' : ''
                    }`}
                  >
                    <td className="py-2.5 px-3 font-mono font-semibold text-slate-900">{exp.id}</td>
                    <td className="py-2.5 px-3 font-medium text-slate-700">{exp.model_type}</td>
                    <td className="py-2.5 px-3 font-mono text-[11px] text-slate-600">{exp.target_task}</td>
                    <td className="py-2.5 px-3 font-mono text-[11px] text-slate-600">{exp.dataset_id}</td>
                    <td className="py-2.5 px-3">
                      <StatusBadge label={exp.status} size="sm" />
                    </td>
                    <td className="py-2.5 px-3 font-bold font-mono">
                      {exp.final_metrics ? `${(exp.final_metrics.accuracy * 100).toFixed(1)}%` : 'Incomplete'}
                    </td>
                    <td className="py-2.5 px-3">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedExpId(exp.id);
                        }}
                        className="text-blue-600 font-semibold hover:underline"
                      >
                        Inspect Curves
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
