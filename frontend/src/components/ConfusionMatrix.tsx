import React from 'react';

interface ConfusionMatrixProps {
  matrix: number[][]; // [[TN, FP], [FN, TP]]
  labels?: [string, string]; // e.g. ['SCZ', 'BD']
}

export const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({
  matrix,
  labels = ['SCZ', 'BD'],
}) => {
  const tn = matrix[0]?.[0] || 0;
  const fp = matrix[0]?.[1] || 0;
  const fn = matrix[1]?.[0] || 0;
  const tp = matrix[1]?.[1] || 0;
  const total = tn + fp + fn + tp;

  const getPercent = (val: number) => (total > 0 ? `${((val / total) * 100).toFixed(1)}%` : '0%');

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-sm font-semibold text-slate-800 uppercase tracking-wider">Confusion Matrix (Held-out Test)</h4>
        <span className="text-xs text-slate-500 font-mono">N = {total} Subjects</span>
      </div>

      <div className="grid grid-cols-[auto_1fr_1fr] gap-2 items-center text-center">
        {/* Header row */}
        <div />
        <div className="text-xs font-semibold text-slate-600 pb-1">Pred: {labels[0]}</div>
        <div className="text-xs font-semibold text-slate-600 pb-1">Pred: {labels[1]}</div>

        {/* Row 1: True Class 0 (SCZ) */}
        <div className="text-xs font-semibold text-slate-600 pr-2 text-right">True: {labels[0]}</div>
        <div className="p-4 rounded-lg bg-emerald-50 border border-emerald-200 flex flex-col justify-center items-center">
          <span className="text-xl font-bold text-emerald-800">{tn}</span>
          <span className="text-xs text-emerald-600 font-medium">TN ({getPercent(tn)})</span>
        </div>
        <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 flex flex-col justify-center items-center">
          <span className="text-xl font-bold text-rose-800">{fp}</span>
          <span className="text-xs text-rose-600 font-medium">FP ({getPercent(fp)})</span>
        </div>

        {/* Row 2: True Class 1 (BD) */}
        <div className="text-xs font-semibold text-slate-600 pr-2 text-right">True: {labels[1]}</div>
        <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 flex flex-col justify-center items-center">
          <span className="text-xl font-bold text-rose-800">{fn}</span>
          <span className="text-xs text-rose-600 font-medium">FN ({getPercent(fn)})</span>
        </div>
        <div className="p-4 rounded-lg bg-emerald-50 border border-emerald-200 flex flex-col justify-center items-center">
          <span className="text-xl font-bold text-emerald-800">{tp}</span>
          <span className="text-xs text-emerald-600 font-medium">TP ({getPercent(tp)})</span>
        </div>
      </div>

      <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
        <span>Sensitivity (Recall): {tp + fn > 0 ? `${((tp / (tp + fn)) * 100).toFixed(1)}%` : 'N/A'}</span>
        <span>Specificity: {tn + fp > 0 ? `${((tn / (tn + fp)) * 100).toFixed(1)}%` : 'N/A'}</span>
      </div>

      <div className="mt-3 p-3 bg-slate-50 rounded-lg text-[11px] text-slate-600 space-y-1">
        <div><strong>False Positives (FP):</strong> Actual {labels[0]} cases incorrectly predicted as {labels[1]}.</div>
        <div><strong>False Negatives (FN):</strong> Actual {labels[1]} cases incorrectly predicted as {labels[0]}.</div>
      </div>
    </div>
  );
};
