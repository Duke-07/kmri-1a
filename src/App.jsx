import React, { useState } from 'react';
import Header from './components/Header.jsx';
import MetricCards from './components/MetricCards.jsx';
import LiveTerminalTab from './components/LiveTerminalTab.jsx';
import CrisisReplayTab from './components/CrisisReplayTab.jsx';
import ConformalCalibrationTab from './components/ConformalCalibrationTab.jsx';
import TacticalBacktestTab from './components/TacticalBacktestTab.jsx';
import ICArtefactTab from './components/ICArtefactTab.jsx';
import PipelineArchitectureTab from './components/PipelineArchitectureTab.jsx';
import { ShieldCheck, Heart, Github, ExternalLink } from './components/Icons.jsx';

export default function App() {
  const [activeTab, setActiveTab] = useState('terminal');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-indigo-500 selection:text-white">
      {/* Top Sticky Header */}
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Container */}
      <main className="flex-1 app-container py-6">
        {/* High-Level Quant KPIs */}
        <MetricCards />

        {/* Tab Content Panels */}
        <div className="transition-opacity duration-200">
          {activeTab === 'terminal' && <LiveTerminalTab />}
          {activeTab === 'crisis' && <CrisisReplayTab />}
          {activeTab === 'calibration' && <ConformalCalibrationTab />}
          {activeTab === 'backtest' && <TacticalBacktestTab />}
          {activeTab === 'artefact' && <ICArtefactTab />}
          {activeTab === 'architecture' && <PipelineArchitectureTab />}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/10 bg-slate-950/80 py-6 text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-200">
              Bayesian Regime Detection Engine
            </span>
            <span className="text-slate-600">•</span>
            <span>Built by Aaryan Dwivedi</span>
          </div>

          <div className="flex items-center gap-4">
            <a
              href="https://github.com/Duke-07/kmri-1a"
              target="_blank"
              rel="noreferrer"
              className="hover:text-indigo-400 transition-colors flex items-center gap-1"
            >
              <Github className="w-3.5 h-3.5" /> GitHub Repository
            </a>
            <span className="text-slate-600">•</span>
            <span className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-emerald-400 font-mono text-[11px]">
              Vercel Edge Ready
            </span>
            <span className="text-slate-600">•</span>
            <span>MIT License</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
