import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';
import { OverviewDashboard } from './pages/OverviewDashboard';
import { DatasetManager } from './pages/DatasetManager';
import { SignalExplorer } from './pages/SignalExplorer';
import { ModelLab } from './pages/ModelLab';
import { ModelEvaluation } from './pages/ModelEvaluation';
import { AnalyzeEEG } from './pages/AnalyzeEEG';
import { Explainability } from './pages/Explainability';
import { AnalysisHistory } from './pages/AnalysisHistory';
import { Settings } from './pages/Settings';
import { ResearchReports } from './pages/ResearchReports';

export function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-50 flex flex-col font-sans text-slate-900">
        <Navbar />
        <div className="flex-1 flex flex-row">
          <Sidebar />
          <main className="flex-1 bg-slate-50/60 overflow-y-auto">
            <Routes>
              {/* 1. Page 1: Home / Prediction */}
              <Route path="/" element={<AnalyzeEEG />} />
              <Route path="/predict" element={<Navigate to="/" replace />} />
              <Route path="/analyze" element={<Navigate to="/" replace />} />

              {/* 2. Page 2: Prediction History */}
              <Route path="/history" element={<AnalysisHistory />} />

              {/* 3. Page 3: Explainability */}
              <Route path="/explainability" element={<Explainability />} />
              <Route path="/evidence" element={<Navigate to="/explainability" replace />} />
              <Route path="/xai" element={<Navigate to="/explainability" replace />} />

              {/* Optional Internal Diagnostics (viva telemetry & audit) */}
              <Route path="/diagnostics" element={<OverviewDashboard />} />
              <Route path="/overview" element={<Navigate to="/diagnostics" replace />} />
              <Route path="/settings" element={<Settings />} />

              {/* Internal research & experiment routes */}
              <Route path="/reports" element={<ResearchReports />} />
              <Route path="/datasets" element={<DatasetManager />} />
              <Route path="/explorer" element={<SignalExplorer />} />
              <Route path="/evaluation" element={<ModelEvaluation />} />
              <Route path="/training" element={<ModelLab />} />

              {/* Fallback */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;
