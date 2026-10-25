import React from 'react';

interface StatusBadgeProps {
  label: string;
  size?: 'sm' | 'md';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ label, size = 'md' }) => {
  const norm = label.toUpperCase();
  const py = size === 'sm' ? 'py-0.5 px-2 text-xs' : 'py-1 px-2.5 text-xs';

  if (norm === 'SCZ' || norm === 'SCHIZOPHRENIA') {
    return (
      <span className={`inline-flex items-center font-medium rounded-full bg-rose-50 text-rose-700 border border-rose-200 ${py}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-rose-500 mr-1.5 animate-pulse" />
        SCZ
      </span>
    );
  }

  if (norm === 'BD' || norm === 'BIPOLAR') {
    return (
      <span className={`inline-flex items-center font-medium rounded-full bg-blue-50 text-blue-700 border border-blue-200 ${py}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-blue-500 mr-1.5" />
        BD
      </span>
    );
  }

  if (norm === 'HC' || norm === 'CONTROL') {
    return (
      <span className={`inline-flex items-center font-medium rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 ${py}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mr-1.5" />
        HC
      </span>
    );
  }

  if (norm === 'UNCERTAIN') {
    return (
      <span className={`inline-flex items-center font-medium rounded-full bg-amber-50 text-amber-700 border border-amber-300 ${py}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-amber-500 mr-1.5" />
        UNCERTAIN
      </span>
    );
  }

  if (norm.includes('SYNTHETIC') || norm.includes('DEMO')) {
    return (
      <span className={`inline-flex items-center font-semibold rounded-md bg-purple-50 text-purple-700 border border-purple-200 ${py}`}>
        [SYNTHETIC DEMO]
      </span>
    );
  }

  if (norm === 'MODEL READY' || norm === 'READY') {
    return (
      <span className={`inline-flex items-center font-bold tracking-wide rounded-full bg-emerald-50 text-emerald-700 border border-emerald-300 ${py}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mr-1.5" />
        MODEL READY
      </span>
    );
  }

  if (norm === 'DATA REQUIRED') {
    return (
      <span className={`inline-flex items-center font-bold tracking-wide rounded-full bg-amber-50 text-amber-800 border border-amber-300 ${py}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-amber-500 mr-1.5" />
        DATA REQUIRED
      </span>
    );
  }

  if (norm === 'MODEL NOT TRAINED') {
    return (
      <span className={`inline-flex items-center font-bold tracking-wide rounded-full bg-amber-50 text-amber-800 border border-amber-200 ${py}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-amber-500 mr-1.5" />
        MODEL NOT TRAINED
      </span>
    );
  }

  if (norm === 'STATUS UNKNOWN') {
    return (
      <span className={`inline-flex items-center font-bold tracking-wide rounded-full bg-rose-50 text-rose-700 border border-rose-300 ${py}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-rose-500 mr-1.5" />
        STATUS UNKNOWN
      </span>
    );
  }

  if (norm === 'COMPLETED') {
    return (
      <span className={`inline-flex items-center font-medium rounded-full bg-teal-50 text-teal-700 border border-teal-200 ${py}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-teal-500 mr-1.5" />
        {label}
      </span>
    );
  }

  if (norm.includes('UNAVAILABLE') || norm === 'FAILED') {
    return (
      <span className={`inline-flex items-center font-medium rounded-full bg-slate-100 text-slate-700 border border-slate-300 ${py}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-slate-400 mr-1.5" />
        {label}
      </span>
    );
  }

  return (
    <span className={`inline-flex items-center font-medium rounded-full bg-slate-100 text-slate-800 border border-slate-200 ${py}`}>
      {label}
    </span>
  );
};
