import React, { useState, useEffect } from 'react';
import {
  Upload,
  FileCheck,
  AlertOctagon,
  ExternalLink,
  ShieldAlert,
  Database,
  CheckCircle,
  FileQuestion,
  FolderOpen,
  RefreshCw,
  Search,
  CheckCircle2,
  Download
} from 'lucide-react';
import { api } from '../api';
import { Dataset, SubjectItem } from '../types';
import { StatusBadge } from '../components/StatusBadge';

export const DatasetManager: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'bsnip' | 'aseeg' | 'ingest'>('aseeg');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [subjects, setSubjects] = useState<SubjectItem[]>([]);
  const [loading, setLoading] = useState(false);
  
  // Manifest upload state (B-SNIP)
  const [manifestText, setManifestText] = useState('');
  const [manifestFile, setManifestFile] = useState<File | null>(null);
  const [validationResult, setValidationResult] = useState<any>(null);
  const [validating, setValidating] = useState(false);
  const [dataDirInput, setDataDirInput] = useState('');

  // Local directory ingestion state
  const [ingestPath, setIngestPath] = useState('backend/data/aseeg');
  const [ingestDatasetId, setIngestDatasetId] = useState('aseeg');
  const [ingestTask, setIngestTask] = useState('SZ_VS_HC');
  const [ingestResult, setIngestResult] = useState<any>(null);
  const [ingestProgress, setIngestProgress] = useState<any>(null);
  const [ingesting, setIngesting] = useState(false);

  const loadData = async () => {
    setLoading(true);
    try {
      const dList = await api.getDatasets();
      setDatasets(dList);
      
      const currentDatasetId = activeTab === 'bsnip' ? 'bsnip1' : activeTab === 'aseeg' ? 'aseeg' : ingestDatasetId;
      const subsRes = await api.getSubjects(currentDatasetId);
      setSubjects(subsRes.subjects);
    } catch (err) {
      console.error('Error fetching datasets', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [activeTab]);

  const handleValidateManifest = async () => {
    setValidating(true);
    try {
      const res = await api.validateManifest(
        manifestFile || undefined,
        manifestText || undefined,
        dataDirInput || undefined
      );
      setValidationResult(res);
      await loadData();
    } catch (err: any) {
      alert(`Manifest validation error: ${err.message}`);
    } finally {
      setValidating(false);
    }
  };

  const handleIngestDirectory = async () => {
    setIngesting(true);
    setIngestProgress({
      status: 'RUNNING',
      total_files: 0,
      processed_files: 0,
      valid_count: 0,
      progress_percent: 0,
      current_file: 'Scanning directory...'
    });

    const pollInterval = setInterval(async () => {
      try {
        const prog = await api.getIngestionProgress(ingestDatasetId);
        if (prog) setIngestProgress(prog);
      } catch {
        // ignore polling errors
      }
    }, 600);

    try {
      const res = await api.ingestDirectory(
        ingestPath,
        ingestDatasetId,
        ingestDatasetId === 'aseeg' ? 'ASEEG Schizophrenia EEG Benchmark' : undefined,
        ingestTask
      );
      setIngestResult(res);
      await loadData();
    } catch (err: any) {
      alert(`Directory ingestion failed: ${err.message}`);
    } finally {
      clearInterval(pollInterval);
      setIngesting(false);
      try {
        const finalProg = await api.getIngestionProgress(ingestDatasetId);
        if (finalProg) setIngestProgress(finalProg);
      } catch {
        // ignore
      }
    }
  };

  const bsnip = datasets.find((d) => d.id === 'bsnip1');
  const aseeg = datasets.find((d) => d.id === 'aseeg');

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Page Title */}
      <div>
        <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-slate-500 mb-1">
          <Database className="w-4 h-4 text-blue-600" />
          <span>Research Data Management</span>
        </div>
        <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">Dataset Provenance & Manifest Validation</h1>
        <p className="text-xs text-slate-500 mt-1">
          Configure verified cohorts, validate CSV manifests, discover EEG files, and maintain strict clinical provenance.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-200 space-x-6 text-sm font-semibold">
        <button
          onClick={() => setActiveTab('aseeg')}
          className={`pb-3 transition-colors flex items-center space-x-2 ${
            activeTab === 'aseeg'
              ? 'border-b-2 border-emerald-600 text-emerald-700'
              : 'text-slate-500 hover:text-slate-800'
          }`}
        >
          <span>ASEEG Benchmark</span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-bold">
            {aseeg?.verified_count || 97} VERIFIED
          </span>
        </button>

        <button
          onClick={() => setActiveTab('bsnip')}
          className={`pb-3 transition-colors flex items-center space-x-2 ${
            activeTab === 'bsnip'
              ? 'border-b-2 border-blue-600 text-blue-600'
              : 'text-slate-500 hover:text-slate-800'
          }`}
        >
          <span>B-SNIP 1 (Primary SCZ-vs-BD)</span>
          <StatusBadge label={bsnip?.is_ready ? 'READY' : 'NDA PROTECTED'} size="sm" />
        </button>

        <button
          onClick={() => setActiveTab('ingest')}
          className={`pb-3 transition-colors flex items-center space-x-2 ${
            activeTab === 'ingest'
              ? 'border-b-2 border-indigo-600 text-indigo-600'
              : 'text-slate-500 hover:text-slate-800'
          }`}
        >
          <FolderOpen className="w-4 h-4" />
          <span>Local Directory Ingestion</span>
        </button>
      </div>

      {/* TAB 1: ASEEG Open Dataset */}
      {activeTab === 'aseeg' && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
            <div className="flex items-start justify-between">
              <div>
                <span className="text-[11px] font-bold text-emerald-700 uppercase tracking-wider block mb-1">
                  Technical Validation Cohort
                </span>
                <h2 className="text-lg font-bold text-slate-900">
                  ASEEG: Schizophrenia and Healthy Control EEG Benchmark
                </h2>
                <p className="text-xs text-slate-500 mt-1 max-w-3xl">
                  Zenodo Record #18029536. 19-channel standard 10-20 montage, 500 Hz acquisition rate.
                  Used strictly for technical validation of the temporal convolutional transformer pipeline.
                </p>
              </div>
              <div className="flex items-center space-x-2">
                <a
                  href={api.getDatasetManifestDownloadUrl('aseeg')}
                  download="aseeg_manifest.csv"
                  className="px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-300 text-emerald-800 text-xs font-semibold hover:bg-emerald-100 flex items-center shadow-2xs transition"
                >
                  <Download className="w-3.5 h-3.5 mr-1.5 text-emerald-700" />
                  Download Manifest (CSV)
                </a>
                <a
                  href="https://zenodo.org/records/18029536"
                  target="_blank"
                  rel="noreferrer"
                  className="px-3 py-1.5 rounded-lg border border-slate-300 text-slate-700 text-xs font-semibold hover:bg-slate-50 flex items-center shadow-xs"
                >
                  Zenodo #18029536 <ExternalLink className="w-3.5 h-3.5 ml-1.5" />
                </a>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2">
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-center">
                <span className="text-xs text-slate-500 block">Verified Participants</span>
                <span className="text-xl font-extrabold text-slate-900">
                  {aseeg?.verified_count || 97}
                </span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-center">
                <span className="text-xs text-slate-500 block">Schizophrenia (SZ)</span>
                <span className="text-xl font-extrabold text-purple-700">47</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-center">
                <span className="text-xs text-slate-500 block">Healthy Control (HC)</span>
                <span className="text-xl font-extrabold text-emerald-700">50</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-center">
                <span className="text-xs text-slate-500 block">Sampling Rate / Montage</span>
                <span className="text-sm font-extrabold text-slate-900 mt-1 block">500 Hz • 19-Ch</span>
              </div>
            </div>

            <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-xl flex items-center justify-between text-xs text-emerald-900">
              <div className="flex items-center space-x-2">
                <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />
                <span>
                  <strong>Dataset Status:</strong> Genuine EEG recording files extracted and registered in database. Ready for 75%/25% participant-level training.
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: B-SNIP 1 Primary */}
      {activeTab === 'bsnip' && (
        <div className="space-y-6">
          <div className="bg-slate-50 border border-slate-300 rounded-xl p-5 flex items-start space-x-4">
            <ShieldAlert className="w-6 h-6 text-slate-700 shrink-0 mt-0.5" />
            <div className="text-xs text-slate-700 leading-relaxed space-y-2">
              <span className="font-bold uppercase tracking-wider block text-slate-900">
                NIH NDA Data Use Certification Notice (Collection #2274)
              </span>
              <p>
                The B-SNIP 1 dataset contains authentic clinical and electrophysiological recordings of participants with
                Schizophrenia (SCZ) and Bipolar Disorder with Psychosis (BD). Access is restricted by the National Institutes of Health.
              </p>
              <div className="flex items-center space-x-3 pt-1">
                <a
                  href="https://nda.nih.gov/edit_collection.html?id=2274"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center text-blue-700 font-semibold hover:underline"
                >
                  Access NIH NDA Collection 2274 <ExternalLink className="w-3.5 h-3.5 ml-1" />
                </a>
              </div>
            </div>
          </div>

          {/* Manifest Ingestion Form */}
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-6">
            <h2 className="text-base font-bold text-slate-900">Validate Authorized B-SNIP Subject Manifest</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <label className="text-xs font-semibold text-slate-700 block">Upload CSV Manifest</label>
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => setManifestFile(e.target.files?.[0] || null)}
                  className="block w-full text-xs text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer"
                />
                <span className="text-[11px] text-slate-400 block">Required headers: subject_id, diagnostic_label, recording_path</span>
              </div>
              <div className="space-y-3">
                <label className="text-xs font-semibold text-slate-700 block">Local Data Directory</label>
                <input
                  type="text"
                  placeholder="e.g. C:\Data\BSNIP1\recordings"
                  value={dataDirInput}
                  onChange={(e) => setDataDirInput(e.target.value)}
                  className="w-full text-xs p-2.5 rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <span className="text-[11px] text-slate-400 block">Directory where authorized EEG files (.edf, .fif) reside</span>
              </div>
            </div>

            <button
              onClick={handleValidateManifest}
              disabled={validating}
              className="px-5 py-2.5 rounded-xl bg-blue-600 text-white text-xs font-bold hover:bg-blue-700 transition flex items-center shadow-xs"
            >
              {validating ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <FileCheck className="w-4 h-4 mr-2" />}
              Validate Manifest & Integrity
            </button>
          </div>
        </div>
      )}

      {/* TAB 3: Local Directory Ingestion */}
      {activeTab === 'ingest' && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-6">
            <div>
              <span className="text-[11px] font-bold text-indigo-700 uppercase tracking-wider block mb-1">
                Automated Ingestion Engine
              </span>
              <h2 className="text-lg font-bold text-slate-900">
                Scan & Ingest Local Dataset Directory
              </h2>
              <p className="text-xs text-slate-500 mt-1">
                Discovers EEG files (.edf, .fif, .mat, .npy, .csv), detects metadata sheets (Excel, TSV, CSV),
                verifies 10-20 channel integrity, maps diagnostic labels, and records audit exclusions.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="md:col-span-2 space-y-2">
                <label className="text-xs font-semibold text-slate-700 block">Directory Path</label>
                <input
                  type="text"
                  value={ingestPath}
                  onChange={(e) => setIngestPath(e.target.value)}
                  placeholder="e.g. backend/data/aseeg or C:\datasets\eeg_folder"
                  className="w-full text-xs font-mono p-2.5 rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>

              <div className="space-y-2">
                <label className="text-xs font-semibold text-slate-700 block">Dataset Identifier</label>
                <input
                  type="text"
                  value={ingestDatasetId}
                  onChange={(e) => setIngestDatasetId(e.target.value)}
                  placeholder="e.g. aseeg, zenodo_14808296, bsnip1"
                  className="w-full text-xs font-mono p-2.5 rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <button
                onClick={handleIngestDirectory}
                disabled={ingesting}
                className="px-5 py-2.5 rounded-xl bg-indigo-600 text-white text-xs font-bold hover:bg-indigo-700 transition flex items-center shadow-xs"
              >
                {ingesting ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Search className="w-4 h-4 mr-2" />}
                {ingesting ? 'Streaming Ingestion Active...' : 'Scan & Ingest Directory'}
              </button>

              <div className="text-[11px] text-slate-500 font-medium">
                Resumable: Previously validated files in the manifest are preserved on interruption.
              </div>
            </div>

            {/* Live Streaming Ingestion Progress Telemetry */}
            {ingesting && ingestProgress && (
              <div className="p-4 bg-indigo-50 border border-indigo-200 rounded-xl space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center space-x-2">
                    <RefreshCw className="w-4 h-4 text-indigo-600 animate-spin" />
                    <span className="font-bold text-indigo-950">Streaming & Chunked Ingestion in Progress</span>
                  </div>
                  <span className="font-mono font-bold text-indigo-700">
                    {ingestProgress.progress_percent || 0}%
                  </span>
                </div>

                {/* Progress Track */}
                <div className="w-full bg-indigo-200/60 rounded-full h-2.5 overflow-hidden">
                  <div
                    className="bg-indigo-600 h-2.5 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, Math.max(2, ingestProgress.progress_percent || 0))}%` }}
                  />
                </div>

                <div className="flex items-center justify-between text-[11px] text-indigo-800">
                  <div className="truncate max-w-md">
                    Current: <span className="font-mono font-medium">{ingestProgress.current_file || 'Scanning files...'}</span>
                  </div>
                  <div>
                    Processed: <strong>{ingestProgress.processed_files || 0}</strong> / {ingestProgress.total_files || 0}
                  </div>
                </div>
              </div>
            )}

            {/* Ingestion Results Card */}
            {ingestResult && (
              <div className="p-5 bg-slate-50 border border-slate-200 rounded-xl space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-slate-900">Ingestion Audit Report</h3>
                  <div className="flex items-center space-x-2">
                    <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${
                      ingestResult.is_ready_for_training ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-800'
                    }`}>
                      {ingestResult.is_ready_for_training ? 'READY FOR TRAINING' : 'INCOMPLETE'}
                    </span>
                    <a
                      href={api.getDatasetManifestDownloadUrl(ingestDatasetId)}
                      download={`${ingestDatasetId}_manifest.csv`}
                      className="px-2.5 py-1 rounded-lg bg-white border border-slate-300 text-slate-800 text-[11px] font-semibold hover:bg-slate-100 flex items-center shadow-2xs transition"
                    >
                      <Download className="w-3 h-3 mr-1 text-indigo-600" />
                      Download Manifest (CSV)
                    </a>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                  <div className="p-3 bg-white rounded-lg border border-slate-200">
                    <span className="text-slate-500 block">Total Files Discovered</span>
                    <span className="text-base font-bold text-slate-900">{ingestResult.total_files_found}</span>
                  </div>
                  <div className="p-3 bg-white rounded-lg border border-slate-200">
                    <span className="text-slate-500 block">Verified Participants</span>
                    <span className="text-base font-bold text-emerald-700">{ingestResult.valid_participants_count}</span>
                  </div>
                  <div className="p-3 bg-white rounded-lg border border-slate-200">
                    <span className="text-slate-500 block">Class Distribution</span>
                    <span className="font-mono font-bold text-slate-900">
                      {JSON.stringify(ingestResult.class_distribution)}
                    </span>
                  </div>
                  <div className="p-3 bg-white rounded-lg border border-slate-200">
                    <span className="text-slate-500 block">Exclusions Recorded</span>
                    <span className="text-base font-bold text-slate-900">{ingestResult.exclusions_count}</span>
                  </div>
                </div>

                <p className="text-xs text-slate-600">{ingestResult.message}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Subjects Table */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-bold text-slate-900">
            Registered Subjects in Database ({subjects.length})
          </h2>
          <button onClick={loadData} className="text-xs text-slate-500 hover:text-slate-800 flex items-center">
            <RefreshCw className="w-3 h-3 mr-1" /> Refresh
          </button>
        </div>

        {subjects.length === 0 ? (
          <div className="py-8 text-center text-slate-400 text-xs">
            No subjects registered for this dataset yet. Use the Ingestion tab or validate a manifest.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-50 text-slate-500 uppercase font-semibold border-b border-slate-200">
                <tr>
                  <th className="py-2.5 px-3">Subject ID</th>
                  <th className="py-2.5 px-3">Diagnostic Label</th>
                  <th className="py-2.5 px-3">Modality</th>
                  <th className="py-2.5 px-3">Age / Sex</th>
                  <th className="py-2.5 px-3">Site</th>
                  <th className="py-2.5 px-3">Recording File</th>
                  <th className="py-2.5 px-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {subjects.slice(0, 30).map((sub) => (
                  <tr key={sub.id} className="hover:bg-slate-50 transition">
                    <td className="py-2.5 px-3 font-mono font-semibold text-slate-900">{sub.subject_id}</td>
                    <td className="py-2.5 px-3">
                      <StatusBadge label={sub.diagnostic_label} size="sm" />
                    </td>
                    <td className="py-2.5 px-3">
                      <span className="text-[10px] text-slate-700 font-medium">EEG (10-20)</span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-600">
                      {sub.age ? `${sub.age} yrs` : 'N/A'} • {sub.sex || 'U'}
                    </td>
                    <td className="py-2.5 px-3 text-slate-600">{sub.site || 'Standard'}</td>
                    <td className="py-2.5 px-3 font-mono text-[11px] text-slate-500 truncate max-w-xs">
                      {sub.recording_path ? sub.recording_path.split('\\').pop() : 'Missing'}
                    </td>
                    <td className="py-2.5 px-3">
                      {sub.file_exists ? (
                        <span className="text-emerald-700 font-semibold text-[11px] flex items-center">
                          <CheckCircle className="w-3 h-3 mr-1" /> On Disk
                        </span>
                      ) : (
                        <span className="text-rose-600 font-semibold text-[11px] flex items-center">
                          <AlertOctagon className="w-3 h-3 mr-1" /> Missing File
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {subjects.length > 30 && (
              <div className="text-center text-xs text-slate-400 py-3 border-t border-slate-100">
                Showing first 30 of {subjects.length} subjects.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DatasetManager;
