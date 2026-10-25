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
              {/* Core User-Facing Pages */}
              <Route path="/" element={<AnalyzeEEG />} />
              <Route path="/predict" element={<Navigate to="/" replace />} />
              <Route path="/analyze" element={<Navigate to="/" replace />} />

              <Route path="/history" element={<AnalysisHistory />} />

              <Route path="/explainability" element={<Explainability />} />
              <Route path="/evidence" element={<Navigate to="/explainability" replace />} />
              <Route path="/xai" element={<Navigate to="/explainability" replace />} />

              {/* Redirect legacy developer/training paths to prediction application */}
              <Route path="/diagnostics" element={<Navigate to="/" replace />} />
              <Route path="/overview" element={<Navigate to="/" replace />} />
              <Route path="/settings" element={<Navigate to="/" replace />} />
              <Route path="/reports" element={<Navigate to="/" replace />} />
              <Route path="/datasets" element={<Navigate to="/" replace />} />
              <Route path="/explorer" element={<Navigate to="/" replace />} />
              <Route path="/evaluation" element={<Navigate to="/" replace />} />
              <Route path="/training" element={<Navigate to="/" replace />} />

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
