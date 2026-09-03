import React, { useState, useEffect } from 'react';
import Header from './components/Header.jsx';
import MetricCards from './components/MetricCards.jsx';
import LiveTerminalTab from './components/LiveTerminalTab.jsx';
import CrisisReplayTab from './components/CrisisReplayTab.jsx';
import ConformalCalibrationTab from './components/ConformalCalibrationTab.jsx';
import TacticalBacktestTab from './components/TacticalBacktestTab.jsx';
import ICArtefactTab from './components/ICArtefactTab.jsx';
import PipelineArchitectureTab from './components/PipelineArchitectureTab.jsx';
import { Github } from './components/Icons.jsx';

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
    <div className="min-h-screen app-shell text-slate-900 flex flex-col antialiased selection:bg-indigo-100 selection:text-indigo-900">
      <Header activeTab={activeTab} setActiveTab={handleTabChange} />

      <main className="flex-1 app-container">
        <MetricCards />

        <div className="tab-panel">
          {activeTab === 'terminal' && <LiveTerminalTab />}
          {activeTab === 'crisis' && <CrisisReplayTab />}
          {activeTab === 'calibration' && <ConformalCalibrationTab />}
          {activeTab === 'backtest' && <TacticalBacktestTab />}
          {activeTab === 'artefact' && <ICArtefactTab />}
          {activeTab === 'architecture' && <PipelineArchitectureTab />}
        </div>
      </main>

      <footer className="border-t border-slate-200/80 bg-white/80 backdrop-blur-sm py-5">
        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span className="font-semibold text-slate-800">Bayesian Regime Detection Engine</span>
            <span className="text-slate-300">·</span>
            <span>Personal quantitative research by <strong className="text-slate-700">Aaryan Dwivedi</strong></span>
          </div>
          <div className="flex items-center gap-3 text-xs text-slate-500">
            <a
              href="https://github.com/Duke-07/kmri-1a"
              target="_blank"
              rel="noreferrer"
              className="hover:text-indigo-600 transition-colors flex items-center gap-1 font-medium text-slate-600"
            >
              <Github className="w-3.5 h-3.5" /> GitHub
            </a>
            <span className="text-slate-300">·</span>
            <span className="px-2 py-0.5 rounded-md bg-indigo-50 border border-indigo-200/80 text-indigo-700 font-mono text-[10px] font-semibold">
              v1.5.0
            </span>
            <span className="text-slate-300">·</span>
            <span>MIT License</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
