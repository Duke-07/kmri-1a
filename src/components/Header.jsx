import React from 'react';
import { Activity, ShieldCheck, Cpu, GitBranch, ExternalLink, Sparkles } from './Icons.jsx';

export default function Header({ activeTab, setActiveTab }) {
  const tabs = [
    { id: 'terminal', label: 'Live Terminal' },
    { id: 'crisis', label: 'Crisis Replay' },
    { id: 'calibration', label: 'Conformal Calibration' },
    { id: 'backtest', label: 'Tactical Backtest' },
    { id: 'artefact', label: 'IC Artefact' },
    { id: 'architecture', label: '12-Stage Pipeline' },
  ];

  return (
    <header className="border-b border-white/10 bg-slate-950/80 backdrop-blur-xl sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand & Identity */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 p-0.5 shadow-lg shadow-indigo-500/20 flex items-center justify-center">
              <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <Activity className="w-5 h-5 text-cyan-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-1.5">
                  Bayesian Regime Engine
                  <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono font-medium">
                    v1.5.0
                  </span>
                </h1>
              </div>
              <p className="text-xs text-slate-400 flex items-center gap-2">
                <span>Indian Equity Market Quantitative Terminal</span>
                <span className="text-slate-600">•</span>
                <a
                  href="https://github.com/Duke-07/kmri-1a"
                  target="_blank"
                  rel="noreferrer"
                  className="text-slate-400 hover:text-indigo-400 transition-colors flex items-center gap-1"
                >
                  Duke-07 <ExternalLink className="w-2.5 h-2.5" />
                </a>
              </p>
            </div>
          </div>

          {/* Status Indicators */}
          <div className="hidden md:flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="text-xs font-mono text-emerald-400 font-medium">
                Vercel Production Ready
              </span>
            </div>

            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 border border-white/5 text-xs text-slate-300 font-mono">
              <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
              <span>90% Conformal Calibrated</span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-1 overflow-x-auto py-2 border-t border-white/5 scrollbar-none">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all duration-150 ${
                  isActive
                    ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 shadow-sm shadow-indigo-500/10'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
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
