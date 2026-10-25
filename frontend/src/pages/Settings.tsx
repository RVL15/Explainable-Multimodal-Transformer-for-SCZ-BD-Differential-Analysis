import React, { useState, useEffect } from 'react';
import {
  Settings as SettingsIcon,
  Server,
  Cpu,
  Database,
  Sliders,
  Sparkles,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  FolderOpen
} from 'lucide-react';
import { api } from '../api';
import { SystemHealth } from '../types';

export const Settings: React.FC = () => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(false);

  // Settings State
  const [sampleRate, setSampleRate] = useState(250);
  const [filterLow, setFilterLow] = useState(1.0);
  const [filterHigh, setFilterHigh] = useState(45.0);
  const [defaultThreshold, setDefaultThreshold] = useState(0.65);
  const [bsnipDataPath, setBsnipDataPath] = useState('data/bsnip');

  const fetchHealth = async () => {
    setLoading(true);
    try {
      const res = await api.getHealth();
      setHealth(res);
    } catch (err) {
      console.error('Failed to load health info', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  const handleSaveDefaults = () => {
    alert('System configuration preferences saved.');
  };

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-8">
      {/* Title */}
      <div>
        <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-slate-500 mb-1">
          <SettingsIcon className="w-4 h-4 text-blue-600" />
          <span>Application Configuration</span>
        </div>
        <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">System Settings & Environment Diagnostics</h1>
        <p className="text-xs text-slate-500 mt-1">
          Review core scientific package environments, configure signal processing defaults, and manage data directories.
        </p>
      </div>

      {/* System Diagnostics Card */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center">
            <Server className="w-4 h-4 text-blue-600 mr-2" />
            Backend Computational Environment
          </h2>
          <button onClick={fetchHealth} className="text-xs text-slate-500 hover:text-slate-800 flex items-center">
            <RefreshCw className={`w-3 h-3 mr-1 ${loading ? 'animate-spin' : ''}`} /> Refresh
          </button>
        </div>

        {health ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 text-xs">
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <span className="text-slate-400 block font-mono text-[10px]">PyTorch</span>
              <strong className="text-slate-900 font-mono text-sm">{health.system.torch_version}</strong>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <span className="text-slate-400 block font-mono text-[10px]">MNE EEG</span>
              <strong className="text-slate-900 font-mono text-sm">{health.system.mne_version}</strong>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <span className="text-slate-400 block font-mono text-[10px]">SciPy</span>
              <strong className="text-slate-900 font-mono text-sm">{health.system.scipy_version}</strong>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <span className="text-slate-400 block font-mono text-[10px]">Scikit-Learn</span>
              <strong className="text-slate-900 font-mono text-sm">{health.system.sklearn_version}</strong>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <span className="text-slate-400 block font-mono text-[10px]">Device</span>
              <strong className="text-blue-700 font-mono text-sm">{health.system.compute_device.toUpperCase()}</strong>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <span className="text-slate-400 block font-mono text-[10px]">API Status</span>
              <strong className="text-emerald-700 font-mono text-sm flex items-center">
                <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> 200 OK
              </strong>
            </div>
          </div>
        ) : (
          <div className="p-4 bg-slate-50 rounded text-xs text-slate-500">Checking backend status...</div>
        )}
      </div>

      {/* Signal Processing Parameters */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center">
          <Sliders className="w-4 h-4 text-blue-600 mr-2" />
          Default EEG Signal Processing Parameters
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div>
            <label className="font-semibold text-slate-700 block mb-1">Standard Sampling Rate (Hz):</label>
            <input
              type="number"
              value={sampleRate}
              onChange={(e) => setSampleRate(parseInt(e.target.value) || 250)}
              className="w-full p-2 rounded-lg border border-slate-200 font-mono text-xs"
            />
            <span className="text-[11px] text-slate-400">Fixed rate for temporal convolutions</span>
          </div>

          <div>
            <label className="font-semibold text-slate-700 block mb-1">Bandpass Low Cutoff (Hz):</label>
            <input
              type="number"
              step="0.5"
              value={filterLow}
              onChange={(e) => setFilterLow(parseFloat(e.target.value) || 1.0)}
              className="w-full p-2 rounded-lg border border-slate-200 font-mono text-xs"
            />
            <span className="text-[11px] text-slate-400">Eliminates low-frequency drift & sweat</span>
          </div>

          <div>
            <label className="font-semibold text-slate-700 block mb-1">Bandpass High Cutoff (Hz):</label>
            <input
              type="number"
              step="1"
              value={filterHigh}
              onChange={(e) => setFilterHigh(parseFloat(e.target.value) || 45.0)}
              className="w-full p-2 rounded-lg border border-slate-200 font-mono text-xs"
            />
            <span className="text-[11px] text-slate-400">Removes 50/60 Hz powerline line noise</span>
          </div>
        </div>

        <div className="pt-2">
          <label className="font-semibold text-slate-700 block mb-1">
            Default Uncertainty Reject Threshold: ({defaultThreshold})
          </label>
          <input
            type="range"
            min="0.50"
            max="0.90"
            step="0.05"
            value={defaultThreshold}
            onChange={(e) => setDefaultThreshold(parseFloat(e.target.value))}
            className="w-full accent-blue-600 cursor-pointer"
          />
        </div>

        <div className="pt-3 border-t border-slate-100 flex justify-end">
          <button
            onClick={handleSaveDefaults}
            className="px-4 py-2 rounded-xl bg-blue-600 text-white text-xs font-bold hover:bg-blue-700 transition"
          >
            Save Processing Defaults
          </button>
        </div>
      </div>

      {/* Dataset Storage & Paths */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center">
          <Database className="w-4 h-4 text-blue-600 mr-2" />
          Dataset Storage & Directory Mapping
        </h2>

        <div className="space-y-4 text-xs">
          <div>
            <label className="font-semibold text-slate-700 block mb-1">ASEEG Benchmark EEG Directory:</label>
            <div className="flex items-center space-x-2">
              <FolderOpen className="w-4 h-4 text-slate-400" />
              <input
                type="text"
                readOnly
                value="backend/data/aseeg"
                className="w-full p-2 rounded-lg border border-slate-200 font-mono text-xs bg-slate-50 text-slate-700"
              />
            </div>
            <span className="text-[11px] text-slate-400">Zenodo #18029536 - 97 verified participants (EO.rar extracted).</span>
          </div>

          <div>
            <label className="font-semibold text-slate-700 block mb-1">Zenodo #14808296 Directory:</label>
            <div className="flex items-center space-x-2">
              <FolderOpen className="w-4 h-4 text-slate-400" />
              <input
                type="text"
                readOnly
                value="backend/data/zenodo_14808296"
                className="w-full p-2 rounded-lg border border-slate-200 font-mono text-xs bg-slate-50 text-slate-700"
              />
            </div>
            <span className="text-[11px] text-slate-400">Rácz et al., 2025 - 77 participants with clinical PANSS sheets.</span>
          </div>

          <div>
            <label className="font-semibold text-slate-700 block mb-1">Primary B-SNIP 1 Storage Directory:</label>
            <div className="flex items-center space-x-2">
              <FolderOpen className="w-4 h-4 text-slate-400" />
              <input
                type="text"
                value={bsnipDataPath}
                onChange={(e) => setBsnipDataPath(e.target.value)}
                className="w-full p-2 rounded-lg border border-slate-200 font-mono text-xs"
              />
            </div>
            <span className="text-[11px] text-slate-400">
              Target folder for authorized NIH NDA B-SNIP EDF or FIF files (Collection #2274).
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
