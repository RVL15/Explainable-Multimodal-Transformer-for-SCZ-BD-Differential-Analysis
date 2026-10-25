import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  History,
  Search,
  Download,
  Trash2,
  Eye,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  HelpCircle,
  FileText,
  Brain,
  ShieldCheck,
  ChevronRight,
  X
} from 'lucide-react';
import { api } from '../api';
import { AnalysisHistoryRecord } from '../types';

export const AnalysisHistory: React.FC = () => {
  const navigate = useNavigate();
  const [records, setRecords] = useState<AnalysisHistoryRecord[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [search, setSearch] = useState<string>('');
  const [taskFilter, setTaskFilter] = useState<string>('ALL');

  // Selected Record Detail Modal
  const [selectedRecord, setSelectedRecord] = useState<AnalysisHistoryRecord | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const taskParam = taskFilter === 'ALL' ? undefined : taskFilter;
      const res = await api.getAnalysisHistory({
        search: search.trim() || undefined,
        target_task: taskParam,
        limit: 50,
      });
      setRecords(res.records || []);
      setTotalCount(res.total_count || 0);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch analysis history.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [taskFilter]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchHistory();
  };

  const handleDelete = async (id: string, filename: string) => {
    if (!window.confirm(`Are you sure you want to delete the record for "${filename}"? This action cannot be undone.`)) {
      return;
    }
    setDeletingId(id);
    try {
      await api.deleteAnalysisRecord(id);
      setRecords(records.filter((r) => r.id !== id));
      setTotalCount((prev) => Math.max(0, prev - 1));
      if (selectedRecord?.id === id) {
        setSelectedRecord(null);
      }
    } catch (err: any) {
      alert(`Failed to delete record: ${err.message}`);
    } finally {
      setDeletingId(null);
    }
  };

  const formatTaskName = (task: string) => {
    if (task === 'SZ_VS_HC' || task === 'SZ_VS_HC_TECH_VAL') return 'Schizophrenia vs Healthy Control (SZ vs HC)';
    if (task === 'SCZ_VS_BD') return 'Schizophrenia vs Bipolar Disorder (SCZ vs BD)';
    return task;
  };

  const getPredictionBadge = (pred: string) => {
    if (pred === 'HEALTHY_CONTROL' || pred === 'HC') {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
          <CheckCircle2 className="w-3 h-3 mr-1" />
          Healthy Control
        </span>
      );
    }
    if (pred === 'UNCERTAIN') {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-amber-50 text-amber-700 border border-amber-200">
          <HelpCircle className="w-3 h-3 mr-1" />
          Uncertain
        </span>
      );
    }
    if (pred === 'BIPOLAR' || pred === 'BD') {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
          <Brain className="w-3 h-3 mr-1" />
          Bipolar Disorder
        </span>
      );
    }
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-rose-50 text-rose-700 border border-rose-200">
        <Brain className="w-3 h-3 mr-1" />
        Schizophrenia
      </span>
    );
  };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-2xl font-bold tracking-tight text-slate-900">Analysis History</h1>
            <span className="px-2.5 py-0.5 text-xs font-semibold bg-slate-100 text-slate-700 rounded-full border border-slate-200">
              {totalCount} records
            </span>
          </div>
          <p className="text-sm text-slate-500 mt-1">
            Searchable audit log of genuine EEG and multimodal inference evaluations conducted on this platform.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <a
            href={api.getHistoryExportCsvUrl()}
            download="eeg_analysis_history.csv"
            className="px-3.5 py-2 bg-white border border-slate-200 text-slate-700 rounded-lg text-xs font-semibold hover:bg-slate-50 transition flex items-center shadow-2xs"
          >
            <Download className="w-3.5 h-3.5 mr-1.5 text-slate-500" />
            Export CSV
          </a>
          <button
            onClick={fetchHistory}
            disabled={loading}
            className="px-3.5 py-2 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-lg text-xs font-semibold hover:bg-indigo-100 transition flex items-center shadow-2xs disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 mr-1.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs flex flex-col md:flex-row gap-3 items-center justify-between">
        <form onSubmit={handleSearchSubmit} className="flex-1 w-full md:max-w-md relative">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by filename or prediction..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2 text-xs bg-slate-50 border border-slate-200 rounded-lg focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition"
          />
        </form>

        <div className="flex items-center space-x-3 w-full md:w-auto">
          <label className="text-xs font-semibold text-slate-500 shrink-0">Classification Task:</label>
          <select
            value={taskFilter}
            onChange={(e) => setTaskFilter(e.target.value)}
            className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-700 font-medium focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition"
          >
            <option value="ALL">All Diagnostic Tasks</option>
            <option value="SZ_VS_HC">Schizophrenia vs Healthy Control (SZ vs HC)</option>
            <option value="SCZ_VS_BD">Schizophrenia vs Bipolar Disorder (SCZ vs BD)</option>
          </select>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-800 flex items-start space-x-3">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <div className="font-bold">Error loading history</div>
            <p className="mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Table of Records */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-2xs overflow-hidden">
        {loading && records.length === 0 ? (
          <div className="p-12 text-center text-slate-400 text-xs flex flex-col items-center justify-center space-y-2">
            <RefreshCw className="w-5 h-5 animate-spin text-indigo-500" />
            <span>Loading genuine analysis records...</span>
          </div>
        ) : records.length === 0 ? (
          <div className="p-12 text-center space-y-4">
            <div className="w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center mx-auto text-slate-400">
              <History className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">No analysis records found</h3>
              <p className="text-xs text-slate-500 max-w-md mx-auto mt-1">
                {search || taskFilter !== 'ALL'
                  ? 'No analysis matches your active filter. Try resetting search criteria.'
                  : 'You have not performed any EEG or multimodal predictions yet. Upload an EEG recording to begin.'}
              </p>
            </div>
            <button
              onClick={() => navigate('/analyze')}
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-xs font-bold hover:bg-indigo-700 transition inline-flex items-center shadow-xs"
            >
              <Brain className="w-3.5 h-3.5 mr-1.5" />
              Analyze EEG Recording
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-600">
              <thead className="bg-slate-50/75 border-b border-slate-200 text-slate-500 font-semibold uppercase tracking-wider text-[10px]">
                <tr>
                  <th className="py-3 px-4">Date / Time</th>
                  <th className="py-3 px-4">Input Filename</th>
                  <th className="py-3 px-4">Task</th>
                  <th className="py-3 px-4">Predicted Class</th>
                  <th className="py-3 px-4">Confidence</th>
                  <th className="py-3 px-4">Quality / Status</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-sans">
                {records.map((r) => {
                  const dateStr = new Date(r.created_at).toLocaleString('en-US', {
                    dateStyle: 'medium',
                    timeStyle: 'short',
                  });
                  return (
                    <tr key={r.id} className="hover:bg-slate-50/60 transition group">
                      <td className="py-3.5 px-4 font-mono text-[11px] text-slate-500 whitespace-nowrap">
                        {dateStr}
                      </td>
                      <td className="py-3.5 px-4">
                        <div className="font-semibold text-slate-900 flex items-center space-x-1.5">
                          <FileText className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                          <span className="truncate max-w-[200px]" title={r.filename}>
                            {r.filename}
                          </span>
                        </div>
                        {r.dataset_id && (
                          <div className="text-[10px] text-slate-400 font-mono mt-0.5">
                            Dataset: {r.dataset_id}
                          </div>
                        )}
                      </td>
                      <td className="py-3.5 px-4">
                        <span className="text-[11px] font-medium text-slate-700">
                          {formatTaskName(r.target_task)}
                        </span>
                      </td>
                      <td className="py-3.5 px-4">{getPredictionBadge(r.predicted_label)}</td>
                      <td className="py-3.5 px-4">
                        <div className="flex items-center space-x-2">
                          <div className="w-16 bg-slate-100 rounded-full h-1.5 overflow-hidden">
                            <div
                              className="bg-indigo-600 h-1.5 rounded-full"
                              style={{ width: `${(r.confidence || 0) * 100}%` }}
                            />
                          </div>
                          <span className="font-mono text-[11px] text-slate-700 font-semibold">
                            {((r.confidence || 0) * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="py-3.5 px-4">
                        {r.is_uncertain ? (
                          <span className="inline-flex items-center text-[10px] font-bold text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                            Uncertain Margin
                          </span>
                        ) : (
                          <span className="inline-flex items-center text-[10px] font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                            Calibrated
                          </span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <div className="flex items-center justify-end space-x-2">
                          <button
                            onClick={() => navigate(`/explainability?analysis_id=${r.id}`)}
                            className="px-2.5 py-1 rounded-md bg-indigo-50 hover:bg-indigo-100 text-indigo-700 text-xs font-semibold flex items-center transition"
                            title="View Explainability & Attributions"
                          >
                            <Eye className="w-3.5 h-3.5 mr-1" />
                            Explain
                          </button>
                          <button
                            onClick={() => setSelectedRecord(r)}
                            className="p-1.5 rounded-md hover:bg-slate-100 text-slate-600 hover:text-slate-900 transition"
                            title="View Full Details"
                          >
                            <FileText className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => handleDelete(r.id, r.filename)}
                            disabled={deletingId === r.id}
                            className="p-1.5 rounded-md hover:bg-rose-50 text-slate-400 hover:text-rose-600 transition disabled:opacity-50"
                            title="Delete Record"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Privacy and Data Retention Notice */}
      <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl flex items-start space-x-3 text-xs text-slate-600">
        <ShieldCheck className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-800">Privacy & Data Retention Policy: </span>
          Only anonymized mathematical inference telemetry, classification outputs, and attribution summaries are persisted in this audit database. Raw binary EEG voltage streams are processed in ephemeral memory during inference and are not permanently archived.
        </div>
      </div>

      {/* Record Details Modal */}
      {selectedRecord && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-2xl w-full border border-slate-200 shadow-xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
            <div className="p-5 border-b border-slate-100 flex items-center justify-between bg-slate-50/70">
              <div>
                <h3 className="font-bold text-base text-slate-900">Inference Audit Detail</h3>
                <p className="text-xs text-slate-500 font-mono mt-0.5">ID: {selectedRecord.id}</p>
              </div>
              <button
                onClick={() => setSelectedRecord(null)}
                className="p-1 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6 space-y-5 text-xs text-slate-700 max-h-[75vh] overflow-y-auto">
              {/* Summary Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                  <div className="text-[10px] text-slate-400 font-semibold uppercase">Prediction</div>
                  <div className="mt-1 font-bold text-slate-900">{selectedRecord.predicted_label}</div>
                </div>
                <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                  <div className="text-[10px] text-slate-400 font-semibold uppercase">Confidence</div>
                  <div className="mt-1 font-bold font-mono text-indigo-600">
                    {((selectedRecord.confidence || 0) * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                  <div className="text-[10px] text-slate-400 font-semibold uppercase">Task</div>
                  <div className="mt-1 font-semibold text-slate-800 text-[11px] truncate">
                    {formatTaskName(selectedRecord.target_task)}
                  </div>
                </div>
                <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                  <div className="text-[10px] text-slate-400 font-semibold uppercase">Uncertainty</div>
                  <div className="mt-1 font-bold">
                    {selectedRecord.is_uncertain ? (
                      <span className="text-amber-600">High Uncertainty</span>
                    ) : (
                      <span className="text-emerald-600">Within Threshold</span>
                    )}
                  </div>
                </div>
              </div>

              {/* Class Probability Distribution */}
              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200/80 space-y-3">
                <div className="font-bold text-slate-900 text-xs">Calibrated Probability Distribution</div>
                <div className="space-y-2">
                  <div>
                    <div className="flex justify-between text-[11px] font-semibold text-slate-700 mb-1">
                      <span>Schizophrenia (SZ)</span>
                      <span className="font-mono">
                        {(((selectedRecord.prob_sz ?? selectedRecord.prob_scz) ?? 0) * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2">
                      <div
                        className="bg-indigo-600 h-2 rounded-full"
                        style={{
                          width: `${((selectedRecord.prob_sz ?? selectedRecord.prob_scz) ?? 0) * 100}%`,
                        }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-[11px] font-semibold text-slate-700 mb-1">
                      <span>
                        {selectedRecord.target_task === 'SZ_VS_HC' || selectedRecord.target_task === 'SZ_VS_HC_TECH_VAL'
                          ? 'Healthy Control (HC)'
                          : 'Bipolar Disorder (BD)'}
                      </span>
                      <span className="font-mono">
                        {(((selectedRecord.prob_hc ?? selectedRecord.prob_bd) ?? 0) * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2">
                      <div
                        className="bg-emerald-500 h-2 rounded-full"
                        style={{
                          width: `${((selectedRecord.prob_hc ?? selectedRecord.prob_bd) ?? 0) * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* File and Execution Telemetry */}
              <div className="space-y-2 border-t border-slate-100 pt-4">
                <div className="font-bold text-slate-900">Input Telemetry</div>
                <div className="grid grid-cols-2 gap-2 text-slate-600 font-mono text-[11px]">
                  <div>Filename: <span className="text-slate-900">{selectedRecord.filename}</span></div>
                  <div>Timestamp: <span className="text-slate-900">{new Date(selectedRecord.created_at).toLocaleString()}</span></div>
                  <div>Evidence Margin: <span className="text-slate-900">{selectedRecord.evidence_margin?.toFixed(4) ?? 'N/A'}</span></div>
                  <div>Model Version: <span className="text-slate-900">{selectedRecord.model_version || 'PyTorch CNN-Transformer v1.0'}</span></div>
                </div>
              </div>

              {/* Attributions if available */}
              {selectedRecord.top_channels && selectedRecord.top_channels.length > 0 && (
                <div className="space-y-2 border-t border-slate-100 pt-4">
                  <div className="font-bold text-slate-900">Salient EEG Channels (Integrated Gradients)</div>
                  <div className="flex flex-wrap gap-1.5">
                    {selectedRecord.top_channels.map((ch, idx) => (
                      <span key={ch} className="px-2 py-1 bg-indigo-50 text-indigo-700 font-mono font-bold text-xs rounded border border-indigo-200">
                        #{idx + 1} {ch}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="p-4 bg-slate-50 border-t border-slate-100 flex items-center justify-between">
              <button
                onClick={() => {
                  setSelectedRecord(null);
                  navigate('/explainability');
                }}
                className="px-3.5 py-1.5 bg-indigo-600 text-white rounded-lg text-xs font-semibold hover:bg-indigo-700 transition flex items-center"
              >
                Open in Explainability
                <ChevronRight className="w-3.5 h-3.5 ml-1" />
              </button>
              <button
                onClick={() => setSelectedRecord(null)}
                className="px-3.5 py-1.5 bg-white border border-slate-200 text-slate-700 rounded-lg text-xs font-semibold hover:bg-slate-50 transition"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalysisHistory;
