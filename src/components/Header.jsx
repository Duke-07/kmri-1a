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
    <header className="sticky top-0 z-50 border-b border-slate-200/80 bg-white/90 backdrop-blur-xl shadow-[0_1px_0_rgba(15,23,42,0.04)]">
      <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8">
        {/* Brand row */}
        <div className="flex items-center justify-between h-[4.25rem]">
          <div className="flex items-center gap-3.5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-600 to-indigo-500 p-[2px] shadow-md shadow-indigo-500/20">
              <div className="w-full h-full bg-white rounded-[10px] flex items-center justify-center">
                <Activity className="w-5 h-5 text-indigo-600" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-[0.9375rem] font-bold tracking-tight text-slate-900">
                  Bayesian Regime Engine
                </h1>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200/80 font-mono font-semibold tracking-wide">
                  v1.5.0
                </span>
              </div>
              <p className="text-[11px] text-slate-500 flex items-center gap-2 mt-0.5">
                <span>Quantitative research by <strong className="text-slate-700 font-semibold">Aaryan Dwivedi</strong></span>
                <span className="text-slate-300">·</span>
                <a
                  href="https://github.com/Duke-07/kmri-1a"
                  target="_blank"
                  rel="noreferrer"
                  className="text-slate-500 hover:text-indigo-600 transition-colors flex items-center gap-1 font-medium"
                >
                  <Github className="w-3 h-3" /> Duke-07 <ExternalLink className="w-2.5 h-2.5 opacity-60" />
                </a>
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 text-[11px] text-slate-600 font-mono">
              <ShieldCheck className="w-3.5 h-3.5 text-indigo-600" />
              <span>90% Conformal Calibrated</span>
            </div>
            <div className="hidden md:flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200/80 text-[11px] font-mono text-emerald-700 font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 pulse-dot"></span>
              <span>5-Regime Inference</span>
            </div>
          </div>
        </div>

        {/* Tab navigation */}
        <div className="pb-3">
          <nav className="tab-nav" role="tablist">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                role="tab"
                aria-selected={activeTab === tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`tab-btn ${activeTab === tab.id ? 'tab-btn--active' : ''}`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>
    </header>
  );
}
