import React from 'react';
import { NavLink, Link } from 'react-router-dom';
import {
  Brain,
  History,
  FileSearch,
  ShieldCheck,
  Server
} from 'lucide-react';

const NAV_ITEMS = [
  { name: 'Prediction', path: '/', icon: Brain, badge: 'Model Ready' },
  { name: 'Prediction History', path: '/history', icon: History, badge: null },
  { name: 'Explainability', path: '/explainability', icon: FileSearch, badge: null },
];

export const Sidebar: React.FC = () => {
  return (
    <aside className="w-64 bg-white border-r border-slate-200 flex flex-col shrink-0 min-h-[calc(100vh-57px)] select-none">
      <div className="p-4 border-b border-slate-100">
        <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-0.5">Prediction Application</div>
        <p className="text-xs text-slate-500">Neuro-oscillatory inference</p>
      </div>

      <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-semibold transition-all ${
                  isActive
                    ? 'bg-indigo-50 text-indigo-700 shadow-2xs font-bold border-l-4 border-indigo-600 pl-2.5'
                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                }`
              }
            >
              <div className="flex items-center space-x-3">
                <Icon className="w-4 h-4 shrink-0" />
                <span>{item.name}</span>
              </div>
              {item.badge && (
                <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-md bg-emerald-100 text-emerald-800">
                  {item.badge}
                </span>
              )}
            </NavLink>
          );
        })}
      </nav>

      <div className="p-4 border-t border-slate-100 bg-slate-50/70 space-y-3">
        <div>
          <div className="flex items-center space-x-2 text-xs font-semibold text-slate-800 mb-1">
            <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0" />
            <span>Research Prediction</span>
          </div>
          <p className="text-[11px] text-slate-500 leading-relaxed">
            SCZ–BD Differential Analysis
          </p>
          <div className="text-[10px] text-slate-400 font-mono mt-0.5">
            16-Ch 10–20 • 250 Hz Standard
          </div>
        </div>

        {/* Discreet diagnostics link for viva demonstration */}
        <div className="pt-2 border-t border-slate-200/60">
          <Link
            to="/diagnostics"
            className="flex items-center text-[11px] text-slate-500 hover:text-slate-800 transition"
          >
            <Server className="w-3.5 h-3.5 mr-1.5 text-slate-400" />
            <span>System Diagnostics</span>
          </Link>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
