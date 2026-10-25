import React, { useState, useEffect } from 'react';
import {
  Activity,
  Sliders,
  Play,
  RotateCcw,
  Download,
  AlertTriangle,
  Layers,
  Sparkles,
  BarChart2
} from 'lucide-react';
import { api } from '../api';
import { SignalResponse, SpectralAnalysis, SubjectItem, PreprocessingOptions } from '../types';
import { WaveformCanvas } from '../components/WaveformCanvas';
import { StatusBadge } from '../components/StatusBadge';

export const SignalExplorer: React.FC = () => {
  const [subjects, setSubjects] = useState<SubjectItem[]>([]);
  const [selectedSubjectId, setSelectedSubjectId] = useState<string>('');
  const [signalData, setSignalData] = useState<SignalResponse | null>(null);
  const [spectralData, setSpectralData] = useState<SpectralAnalysis | null>(null);
  const [loading, setLoading] = useState(false);

  // Preprocessing Options State
  const [options, setOptions] = useState<PreprocessingOptions>({
    bandpass_enabled: true,
    low_freq_hz: 1.0,
    high_freq_hz: 45.0,
    rereference_average: true,
    normalize_channels: true,
    window_duration_sec: 2.0,
    rejection_threshold_uv: 150.0,
  });

  const [selectedChannels, setSelectedChannels] = useState<string[]>([]);
  const [amplitudeScale, setAmplitudeScale] = useState<number>(1.0);
  const [showProcessed, setShowProcessed] = useState<boolean>(true);

  // Fetch available subjects
  useEffect(() => {
    const fetchSubjects = async () => {
      try {
        const explainable = await api.getExplainableSubjects();
        if (explainable && explainable.length > 0) {
          setSubjects(explainable);
          setSelectedSubjectId(explainable[0].subject_id);
          return;
        }
        const res = await api.getSubjects('aseeg');
        setSubjects(res.subjects);
        if (res.subjects.length > 0) {
          setSelectedSubjectId(res.subjects[0].subject_id);
        }
      } catch (err) {
        console.error('Failed to load subjects', err);
      }
    };
    fetchSubjects();
  }, []);

  // Fetch signal when selectedSubjectId changes
  const loadSignal = async () => {
    if (!selectedSubjectId) return;
    setLoading(true);
    try {
      const sig = await api.preprocessSignal(selectedSubjectId, options, 5.0, 16);
      setSignalData(sig);
      if (selectedChannels.length === 0) {
        setSelectedChannels(sig.channels.slice(0, 8)); // select first 8 by default
      }

      const spec = await api.getSpectral(selectedSubjectId, 8);
      setSpectralData(spec);
    } catch (err: any) {
      alert(`Error loading EEG signal: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedSubjectId) {
      loadSignal();
    }
  }, [selectedSubjectId]);

  const toggleChannel = (ch: string) => {
    if (selectedChannels.includes(ch)) {
      if (selectedChannels.length > 1) {
        setSelectedChannels(selectedChannels.filter((c) => c !== ch));
      }
    } else {
      setSelectedChannels([...selectedChannels, ch]);
    }
  };

  const currentSubject = subjects.find((s) => s.subject_id === selectedSubjectId);

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-slate-500 mb-1">
            <Activity className="w-4 h-4 text-blue-600" />
            <span>EEG Signal Exploration & Preprocessing</span>
          </div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">Interactive Multi-Channel Waveforms & Welch PSD</h1>
          <p className="text-xs text-slate-500 mt-1">
            Zero-phase 1-45 Hz Butterworth bandpass filtering, Common Average Referencing (CAR), and 150 µV artifact rejection.
          </p>
        </div>

        {/* Subject dropdown */}
        <div className="flex items-center space-x-3 bg-white p-2 rounded-xl border border-slate-200 shadow-2xs">
          <label className="text-xs font-semibold text-slate-600">Subject:</label>
          <select
            value={selectedSubjectId}
            onChange={(e) => setSelectedSubjectId(e.target.value)}
            className="text-xs font-mono font-semibold px-3 py-1.5 rounded-lg border border-slate-200 bg-slate-50 text-slate-800 focus:outline-none focus:border-blue-600"
          >
            {subjects.map((sub) => (
              <option key={sub.subject_id} value={sub.subject_id}>
                {sub.subject_id} ({sub.diagnostic_label})
              </option>
            ))}
          </select>
          {currentSubject && (
            <StatusBadge label={currentSubject.diagnostic_label} size="sm" />
          )}
        </div>
      </div>

      {/* Recording Metadata & Input Panel */}
      {currentSubject && (
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-3">
            Recording Metadata & Acquisition Specifications
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 text-xs">
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-[10px] text-slate-400 font-semibold block uppercase">Participant ID</span>
              <strong className="text-slate-900 font-mono text-[11px] mt-0.5 block truncate">
                {currentSubject.subject_id}
              </strong>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-[10px] text-slate-400 font-semibold block uppercase">Dataset</span>
              <strong className="text-slate-900 text-[11px] mt-0.5 block truncate">
                {currentSubject.dataset_id || 'ASEEG'}
              </strong>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-[10px] text-slate-400 font-semibold block uppercase">Diagnosis</span>
              <strong className="text-indigo-700 text-[11px] mt-0.5 block">
                {currentSubject.diagnostic_label}
              </strong>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-[10px] text-slate-400 font-semibold block uppercase">Sampling Rate</span>
              <strong className="text-slate-900 font-mono text-[11px] mt-0.5 block">
                250 Hz
              </strong>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-[10px] text-slate-400 font-semibold block uppercase">Channels</span>
              <strong className="text-slate-900 font-mono text-[11px] mt-0.5 block">
                16 (10–20 Standard)
              </strong>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-[10px] text-slate-400 font-semibold block uppercase">Signal Units</span>
              <strong className="text-slate-900 font-mono text-[11px] mt-0.5 block">
                Microvolts (µV)
              </strong>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-[10px] text-slate-400 font-semibold block uppercase">File Status</span>
              <strong className="text-emerald-700 font-mono text-[11px] mt-0.5 block">
                Validated
              </strong>
            </div>
          </div>
        </div>
      )}

      {/* Preprocessing Controls Drawer */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Sliders className="w-4 h-4 text-blue-600" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800">Signal Processing Configuration</h2>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowProcessed(!showProcessed)}
              className={`text-xs px-3 py-1.5 rounded-lg font-semibold transition ${
                showProcessed ? 'bg-sky-100 text-sky-800 border border-sky-300' : 'bg-slate-100 text-slate-600'
              }`}
            >
              {showProcessed ? 'Showing Processed' : 'Showing Raw Only'}
            </button>
            <button
              onClick={loadSignal}
              disabled={loading}
              className="px-4 py-1.5 rounded-lg bg-blue-600 text-white text-xs font-bold hover:bg-blue-700 transition flex items-center shadow-xs"
            >
              <Play className="w-3.5 h-3.5 mr-1.5" />
              {loading ? 'Processing...' : 'Apply Filters'}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          {/* Bandpass */}
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
            <label className="flex items-center space-x-2 font-semibold text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={options.bandpass_enabled}
                onChange={(e) => setOptions({ ...options, bandpass_enabled: e.target.checked })}
                className="rounded text-blue-600"
              />
              <span>1 - 45 Hz Bandpass Filter</span>
            </label>
            <div className="flex items-center space-x-2">
              <span className="text-slate-500 font-mono">Low:</span>
              <input
                type="number"
                step="0.5"
                min="0.1"
                max="10"
                value={options.low_freq_hz}
                onChange={(e) => setOptions({ ...options, low_freq_hz: parseFloat(e.target.value) || 1.0 })}
                className="w-16 px-2 py-1 rounded bg-white border border-slate-200 text-xs font-mono"
              />
              <span className="text-slate-500 font-mono">High:</span>
              <input
                type="number"
                step="1"
                min="20"
                max="50"
                value={options.high_freq_hz}
                onChange={(e) => setOptions({ ...options, high_freq_hz: parseFloat(e.target.value) || 45.0 })}
                className="w-16 px-2 py-1 rounded bg-white border border-slate-200 text-xs font-mono"
              />
            </div>
          </div>

          {/* Average Rereference (CAR) */}
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
            <label className="flex items-center space-x-2 font-semibold text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={options.rereference_average}
                onChange={(e) => setOptions({ ...options, rereference_average: e.target.checked })}
                className="rounded text-blue-600"
              />
              <span>Common Average Reference (CAR)</span>
            </label>
            <p className="text-[11px] text-slate-500">
              Subtracts instantaneous spatial mean across all electrodes to remove shared noise.
            </p>
          </div>

          {/* Normalization */}
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
            <label className="flex items-center space-x-2 font-semibold text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={options.normalize_channels}
                onChange={(e) => setOptions({ ...options, normalize_channels: e.target.checked })}
                className="rounded text-blue-600"
              />
              <span>Channel Z-Score Normalization</span>
            </label>
            <p className="text-[11px] text-slate-500">
              Standardizes variance per channel to zero-mean unit-variance.
            </p>
          </div>

          {/* Artifact Rejection Threshold */}
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
            <div className="flex justify-between font-semibold text-slate-700">
              <span>Artifact Rejection:</span>
              <span className="font-mono text-blue-600">{options.rejection_threshold_uv} µV</span>
            </div>
            <input
              type="range"
              min="80"
              max="250"
              step="10"
              value={options.rejection_threshold_uv}
              onChange={(e) => setOptions({ ...options, rejection_threshold_uv: parseFloat(e.target.value) })}
              className="w-full accent-blue-600 cursor-pointer"
            />
            <p className="text-[11px] text-slate-500">
              Rejects windows exceeding peak-to-peak amplitude threshold.
            </p>
          </div>
        </div>
      </div>

      {/* Artifact Rejection Summary Bar */}
      {signalData && signalData.total_windows !== undefined && (
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs flex flex-wrap items-center justify-between gap-4 text-xs">
          <div className="flex items-center space-x-6">
            <div>
              <span className="text-slate-500 block">Total Evaluated Windows</span>
              <strong className="text-slate-900 font-mono text-sm">{signalData.total_windows}</strong>
            </div>
            <div>
              <span className="text-slate-500 block">Clean Windows Passed</span>
              <strong className="text-emerald-700 font-mono text-sm">
                {(signalData.total_windows || 0) - (signalData.rejected_windows || 0)}
              </strong>
            </div>
            <div>
              <span className="text-slate-500 block">Artifact Windows Rejected</span>
              <strong className="text-rose-600 font-mono text-sm">{signalData.rejected_windows || 0}</strong>
            </div>
            <div>
              <span className="text-slate-500 block">Rejection Ratio</span>
              <strong className="text-amber-700 font-mono text-sm">
                {((signalData.rejection_ratio || 0) * 100).toFixed(1)}%
              </strong>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-slate-500 font-medium">Amplitude Zoom:</span>
            <button
              onClick={() => setAmplitudeScale(Math.max(0.5, amplitudeScale - 0.25))}
              className="px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 font-bold"
            >
              -
            </button>
            <span className="font-mono px-1 font-bold">{amplitudeScale.toFixed(2)}x</span>
            <button
              onClick={() => setAmplitudeScale(Math.min(3.0, amplitudeScale + 0.25))}
              className="px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 font-bold"
            >
              +
            </button>
          </div>
        </div>
      )}

      {/* Channel Selector Chips */}
      {signalData && (
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs space-y-2">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
            <span>Electrode Channels (10-20 Montage Subset)</span>
            <div className="space-x-2">
              <button
                onClick={() => setSelectedChannels(signalData.channels)}
                className="text-blue-600 hover:underline"
              >
                Select All
              </button>
              <button
                onClick={() => setSelectedChannels(signalData.channels.slice(0, 4))}
                className="text-slate-500 hover:underline"
              >
                Frontal Only
              </button>
            </div>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {signalData.channels.map((ch) => {
              const active = selectedChannels.includes(ch);
              return (
                <button
                  key={ch}
                  onClick={() => toggleChannel(ch)}
                  className={`text-xs px-2.5 py-1 rounded-md font-mono font-medium transition ${
                    active
                      ? 'bg-blue-600 text-white shadow-2xs'
                      : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  {ch}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Interactive Waveform Canvas */}
      {signalData && (
        <div className="space-y-2">
          <WaveformCanvas
            channels={signalData.waveforms}
            timeVector={signalData.time_vector}
            selectedChannels={selectedChannels}
            showProcessed={showProcessed}
            amplitudeScale={amplitudeScale}
            height={520}
          />
        </div>
      )}

      {/* Spectral Analysis: Welch PSD and Canonical Frequency Bands */}
      {spectralData && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Canonical Frequency Bands Distribution */}
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-800">
                Mean Frequency Band Power Distribution
              </h3>
              <span className="text-xs text-slate-500 font-mono">0.5 - 45.0 Hz</span>
            </div>

            <div className="space-y-3">
              {[
                { band: 'Delta (0.5 - 4.0 Hz)', key: 'Delta', color: 'bg-indigo-500' },
                { band: 'Theta (4.0 - 8.0 Hz)', key: 'Theta', color: 'bg-cyan-500' },
                { band: 'Alpha (8.0 - 13.0 Hz)', key: 'Alpha', color: 'bg-emerald-500' },
                { band: 'Beta (13.0 - 30.0 Hz)', key: 'Beta', color: 'bg-amber-500' },
                { band: 'Gamma (30.0 - 45.0 Hz)', key: 'Gamma', color: 'bg-rose-500' },
              ].map((item) => {
                const relVal = spectralData.mean_band_distribution[item.key] || 0;
                return (
                  <div key={item.key} className="space-y-1">
                    <div className="flex justify-between text-xs text-slate-700">
                      <span className="font-semibold">{item.band}</span>
                      <span className="font-mono font-bold">{(relVal * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden">
                      <div
                        className={`h-full ${item.color} rounded-full transition-all duration-500`}
                        style={{ width: `${Math.min(100, relVal * 100)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            <p className="text-[11px] text-slate-500 pt-2 border-t border-slate-100">
              *Relative band power computed via Simpson integration over Welch's power spectral density.
            </p>
          </div>

          {/* Top Channel Power Breakdown */}
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-800">
              Electrode Spectral Densities (Welch's Method)
            </h3>
            <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
              {spectralData.channels_spectral.slice(0, 6).map((ch) => (
                <div key={ch.channel} className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs space-y-1">
                  <div className="flex justify-between font-bold text-slate-800 font-mono">
                    <span>{ch.channel}</span>
                    <span className="text-slate-500 text-[11px]">
                      Alpha: {((ch.band_powers.find((b) => b.band === 'Alpha')?.relative_power || 0) * 100).toFixed(1)}% •{' '}
                      Theta: {((ch.band_powers.find((b) => b.band === 'Theta')?.relative_power || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex space-x-1 h-3 rounded bg-slate-200 overflow-hidden">
                    {ch.band_powers.map((b) => (
                      <div
                        key={b.band}
                        title={`${b.band}: ${(b.relative_power * 100).toFixed(1)}%`}
                        className={`h-full ${
                          b.band === 'Delta' ? 'bg-indigo-500' :
                          b.band === 'Theta' ? 'bg-cyan-500' :
                          b.band === 'Alpha' ? 'bg-emerald-500' :
                          b.band === 'Beta' ? 'bg-amber-500' : 'bg-rose-500'
                        }`}
                        style={{ width: `${b.relative_power * 100}%` }}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
