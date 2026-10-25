import React, { useEffect, useState } from 'react';
import { Server, CheckCircle2, AlertCircle, Database } from 'lucide-react';
import { api } from '../api';
import { SystemHealth } from '../types';

export const Navbar: React.FC = () => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [isOnline, setIsOnline] = useState<boolean>(false);

  useEffect(() => {
    const check = async () => {
      try {
        const res = await api.getHealth();
        setHealth(res);
        setIsOnline(true);
      } catch {
        setIsOnline(false);
      }
    };
    check();
    const interval = setInterval(check, 10000);
    return () => clearInterval(interval);
  }, []);

  const bsnipReady = health?.datasets_status.find((d) => d.id === 'bsnip1')?.is_ready;

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-30 px-6 py-3 flex items-center justify-between shadow-2xs">
      <div className="flex items-center space-x-3">
        <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold text-sm shadow-xs">
          Ψ
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-base font-bold text-slate-900 tracking-tight">
              Explainable Multimodal Transformer
            </h1>
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 font-mono font-medium">
              EEG Research Platform
            </span>
          </div>
          <p className="text-xs text-slate-500">
            Neuropsychiatric Differential Analysis • Multimodal Transformer Architecture
          </p>
        </div>
      </div>

      <div className="flex items-center space-x-3">
        {/* B-SNIP Data Readiness pill */}
        <div className={`hidden sm:flex items-center text-xs px-2.5 py-1 rounded-full border ${
          bsnipReady
            ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
            : 'bg-slate-100 text-slate-600 border-slate-200'
        }`}>
          {bsnipReady ? (
            <CheckCircle2 className="w-3 h-3 mr-1 text-emerald-600" />
          ) : (
            <AlertCircle className="w-3 h-3 mr-1 text-slate-500" />
          )}
          <span>B-SNIP 1: {bsnipReady ? 'Verified' : 'Access Restricted (NDA)'}</span>
        </div>

        {/* Backend health status */}
        <div className="flex items-center text-xs px-2.5 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-700">
          <Server className="w-3 h-3 mr-1.5 text-slate-500" />
          <span className={`w-2 h-2 rounded-full mr-1.5 ${isOnline ? 'bg-emerald-500' : 'bg-rose-500'}`} />
          <span>{isOnline ? `FastAPI (${health?.system.compute_device.toUpperCase() || 'CPU'})` : 'Offline'}</span>
        </div>
      </div>
    </header>
  );
};
