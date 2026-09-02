import React, { useState, useEffect } from 'react';
import Header from './components/Header.jsx';
import MetricCards from './components/MetricCards.jsx';
import LiveTerminalTab from './components/LiveTerminalTab.jsx';
import CrisisReplayTab from './components/CrisisReplayTab.jsx';
import ConformalCalibrationTab from './components/ConformalCalibrationTab.jsx';
import TacticalBacktestTab from './components/TacticalBacktestTab.jsx';
import ICArtefactTab from './components/ICArtefactTab.jsx';
import PipelineArchitectureTab from './components/PipelineArchitectureTab.jsx';
import { ShieldCheck, Github, ExternalLink } from './components/Icons.jsx';

export default function App() {
  const getInitialTab = () => {
    const hash = window.location.hash.replace('#', '');
    const validTabs = ['terminal', 'crisis', 'calibration', 'backtest', 'artefact', 'architecture'];
    return validTabs.includes(hash) ? hash : 'terminal';
  };

  const [activeTab, setActiveTab] = useState(getInitialTab);

  const handleTabChange = (tabId) => {
    setActiveTab(tabId);
    window.location.hash = tabId;
  };

  useEffect(() => {
    const onHashChange = () => {
      const hash = window.location.hash.replace('#', '');
      const validTabs = ['terminal', 'crisis', 'calibration', 'backtest', 'artefact', 'architecture'];
      if (validTabs.includes(hash)) {
        setActiveTab(hash);
      }
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col antialiased selection:bg-indigo-100 selection:text-indigo-900">
      {/* Top Sticky Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={handleTabChange}
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

      {/* Footer (Personal project attribution, non-misleading transparency) */}
      <footer className="border-t border-slate-200 bg-white py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-bold text-slate-800">
              Bayesian Regime Detection Engine
            </span>
            <span className="text-slate-300">•</span>
            <span>Personal Quantitative Research by <strong>Aaryan Dwivedi</strong></span>
          </div>

          <div className="flex items-center gap-4">
            <a
              href="https://github.com/Duke-07/kmri-1a"
              target="_blank"
              rel="noreferrer"
              className="hover:text-indigo-600 transition-colors flex items-center gap-1 font-medium text-slate-600"
            >
              <Github className="w-3.5 h-3.5" /> GitHub
            </a>
            <span className="text-slate-300">•</span>
            <span className="px-2 py-0.5 rounded bg-slate-100 border border-slate-200 text-indigo-700 font-mono text-[11px]">
              Release v1.5.0
            </span>
            <span className="text-slate-300">•</span>
            <span>MIT License</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
