import React from 'react';
import { Activity, ShieldCheck, ExternalLink, Github } from './Icons.jsx';

export default function Header({ activeTab, setActiveTab }) {
  const tabs = [
    { id: 'terminal', label: 'Live Terminal' },
    { id: 'crisis', label: 'Crisis Replay' },
    { id: 'calibration', label: 'Conformal Calibration' },
    { id: 'backtest', label: 'Tactical Backtest' },
    { id: 'artefact', label: 'IC Artefact' },
    { id: 'architecture', label: '14-Stage Pipeline' },
  ];

  return (
    <header className="border-b border-slate-200 bg-white/95 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand & Identity */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-cyan-500 p-0.5 shadow-md shadow-indigo-500/10 flex items-center justify-center">
              <div className="w-full h-full bg-white rounded-[10px] flex items-center justify-center">
                <Activity className="w-5 h-5 text-indigo-600" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold tracking-tight text-slate-900 flex items-center gap-1.5">
                  Bayesian Regime Engine
                  <span className="text-[11px] px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200 font-mono font-semibold">
                    v1.5.0
                  </span>
                </h1>
              </div>
              <p className="text-xs text-slate-500 flex items-center gap-2">
                <span>Personal quantitative research by <strong>Aaryan Dwivedi</strong></span>
                <span className="text-slate-300">•</span>
                <a
                  href="https://github.com/Duke-07/kmri-1a"
                  target="_blank"
                  rel="noreferrer"
                  className="text-slate-600 hover:text-indigo-600 transition-colors flex items-center gap-1 font-medium"
                >
                  <Github className="w-3 h-3" /> Duke-07 <ExternalLink className="w-2.5 h-2.5" />
                </a>
              </p>
            </div>
          </div>

          {/* Controls & Badges (Clean light theme only, no dark mode toggle) */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 border border-slate-200 text-xs text-slate-700 font-mono">
              <ShieldCheck className="w-3.5 h-3.5 text-indigo-600" />
              <span>90% Conformal Calibrated</span>
            </div>

            <div className="hidden md:flex items-center gap-2 px-2.5 py-1 rounded-md bg-emerald-50 border border-emerald-200 text-xs font-mono text-emerald-700">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>5-Regime Inference</span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-1 overflow-x-auto py-2 border-t border-slate-100 scrollbar-none">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all duration-150 ${
                  isActive
                    ? 'bg-indigo-50 text-indigo-700 border border-indigo-200 shadow-sm font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
}
