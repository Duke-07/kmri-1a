import React, { useState, useEffect } from 'react';
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
  const [theme, setTheme] = useState('light'); // Light mode as default

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
      document.body.className = 'bg-slate-950 text-slate-100 antialiased selection:bg-indigo-500 selection:text-white';
    } else {
      root.classList.remove('dark');
      document.body.className = 'bg-slate-50 text-slate-900 antialiased selection:bg-indigo-100 selection:text-indigo-900';
    }
  }, [theme]);

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 flex flex-col transition-colors">
      {/* Top Sticky Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        theme={theme}
        setTheme={setTheme}
      />

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

      {/* Footer (Personal project attribution, zero corporate fluff) */}
      <footer className="border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 py-6 text-xs text-slate-500 dark:text-slate-400 transition-colors">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-bold text-slate-800 dark:text-slate-200">
              Bayesian Regime Detection Engine
            </span>
            <span className="text-slate-300 dark:text-slate-700">•</span>
            <span>Personal Research Project by <strong>Aaryan Dwivedi</strong></span>
          </div>

          <div className="flex items-center gap-4">
            <a
              href="https://github.com/Duke-07/kmri-1a"
              target="_blank"
              rel="noreferrer"
              className="hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors flex items-center gap-1 font-medium"
            >
              <Github className="w-3.5 h-3.5" /> GitHub
            </a>
            <span className="text-slate-300 dark:text-slate-700">•</span>
            <span className="px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-indigo-700 dark:text-indigo-400 font-mono text-[11px]">
              Production Release v1.5.0
            </span>
            <span className="text-slate-300 dark:text-slate-700">•</span>
            <span>MIT License</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
