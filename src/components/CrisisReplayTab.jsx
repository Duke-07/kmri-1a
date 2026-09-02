import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, AlertOctagon, ChevronRight, ShieldAlert, Sparkles, TrendingDown } from './Icons.jsx';
import { CRISIS_SCENARIOS, REGIMES } from '../engine/data';

export default function CrisisReplayTab() {
  const [selectedCrisisId, setSelectedCrisisId] = useState('covid_2020');
  const [currentStepIdx, setCurrentStepIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const crisis = CRISIS_SCENARIOS.find((c) => c.id === selectedCrisisId) || CRISIS_SCENARIOS[0];
  const currentStep = crisis.timeline[currentStepIdx] || crisis.timeline[0];

  // Auto-play timeline
  useEffect(() => {
    let interval = null;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentStepIdx((prev) => {
          if (prev >= crisis.timeline.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 2500);
    }
    return () => clearInterval(interval);
  }, [isPlaying, crisis.timeline.length]);

  const handleSelectCrisis = (id) => {
    setSelectedCrisisId(id);
    setCurrentStepIdx(0);
    setIsPlaying(false);
  };

  return (
    <div className="space-y-6">
      {/* Crisis Scenario Selector Pills */}
      <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-none">
        {CRISIS_SCENARIOS.map((c) => {
          const isSelected = c.id === selectedCrisisId;
          return (
            <button
              key={c.id}
              onClick={() => handleSelectCrisis(c.id)}
              className={`px-4 py-2 rounded-xl text-xs font-medium whitespace-nowrap transition-all flex items-center gap-2 ${
                isSelected
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-sm shadow-rose-500/10'
                  : 'bg-slate-900/60 text-slate-400 border border-white/5 hover:border-white/10 hover:text-slate-200'
              }`}
            >
              <AlertOctagon className="w-3.5 h-3.5 text-rose-400" />
              <span>{c.name}</span>
            </button>
          );
        })}
      </div>

      {/* Scenario Hero Card */}
      <div className="glass-panel p-5 rounded-2xl border border-rose-500/20 bg-gradient-to-r from-slate-900/90 via-slate-900/60 to-rose-950/30">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-rose-400 mb-1">
              <span>{crisis.period}</span>
              <span>•</span>
              <span>BOCPD Changepoint: {crisis.bocpdConfidence}</span>
              <span>•</span>
              <span>Early Warning Lead: {crisis.leadDays}</span>
            </div>
            <h2 className="text-2xl font-bold tracking-tight text-white">{crisis.name}</h2>
            <p className="text-xs text-slate-300 mt-1">{crisis.headline}</p>
          </div>

          <div className="flex items-center gap-3">
            <div className="px-3.5 py-2 rounded-xl bg-slate-900/80 border border-white/10 text-right font-mono text-xs">
              <div className="text-[10px] text-slate-400">PEAK INDIA VIX</div>
              <div className="text-rose-400 font-bold text-sm">{crisis.keyFeatures.vix}</div>
            </div>
            <div className="px-3.5 py-2 rounded-xl bg-slate-900/80 border border-white/10 text-right font-mono text-xs">
              <div className="text-[10px] text-slate-400">MAX DRAWDOWN</div>
              <div className="text-rose-400 font-bold text-sm">{crisis.keyFeatures.drawdown}</div>
            </div>
            <div className="px-3.5 py-2 rounded-xl bg-slate-900/80 border border-white/10 text-right font-mono text-xs">
              <div className="text-[10px] text-slate-400">TDA L2 NORM</div>
              <div className="text-amber-400 font-bold text-sm">{crisis.keyFeatures.tdaNorm}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Interactive Playback Controller */}
      <div className="glass-panel p-4 rounded-xl border border-white/5 bg-slate-900/60 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition-all shadow-md shadow-indigo-600/20"
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{isPlaying ? 'Pause Replay' : 'Play Timeline'}</span>
          </button>

          <button
            onClick={() => {
              setCurrentStepIdx(0);
              setIsPlaying(false);
            }}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 text-xs transition-colors"
            title="Reset replay to beginning"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          <span className="text-xs font-mono text-slate-400">
            Step {currentStepIdx + 1} of {crisis.timeline.length}
          </span>
        </div>

        {/* Scrubber step buttons */}
        <div className="flex items-center gap-1 overflow-x-auto max-w-full">
          {crisis.timeline.map((step, idx) => (
            <button
              key={idx}
              onClick={() => {
                setCurrentStepIdx(idx);
                setIsPlaying(false);
              }}
              className={`px-2.5 py-1 rounded-md text-[11px] font-mono transition-all ${
                currentStepIdx === idx
                  ? 'bg-rose-500 text-white font-bold'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white'
              }`}
            >
              {step.date.slice(5)}
            </button>
          ))}
        </div>
      </div>

      {/* Step Replay Inspection Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left (7 cols): Current Date Snapshot & Action */}
        <div className="lg:col-span-7 glass-panel p-5 rounded-2xl border border-white/5 bg-slate-900/60 space-y-4">
          <div className="flex items-center justify-between border-b border-white/5 pb-3">
            <div>
              <span className="text-[10px] uppercase font-mono tracking-wider text-slate-500">
                REPLAY DATE SNAPSHOT
              </span>
              <h3 className="text-xl font-bold font-mono text-white flex items-center gap-2">
                {currentStep.date}
                <span className="text-xs px-2 py-0.5 rounded bg-white/10 text-slate-300 font-normal">
                  NIFTY: {currentStep.nifty.toLocaleString('en-IN')}
                </span>
              </h3>
            </div>
            <div className="text-right">
              <span className="text-[10px] uppercase font-mono tracking-wider text-slate-500">
                RISK-OFF POSTERIOR
              </span>
              <div className="text-lg font-bold font-mono text-rose-400">
                {(currentStep.pRiskOff * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Engine Action Card */}
          <div className="p-4 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-xs">
            <div className="flex items-center gap-1.5 text-indigo-300 font-semibold mb-1">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Bayesian Engine Autonomous Response</span>
            </div>
            <p className="text-slate-200 font-mono text-[13px] leading-relaxed">
              "{currentStep.action}"
            </p>
          </div>

          {/* Probability Breakdown for this date */}
          <div className="space-y-3 pt-2">
            <h4 className="text-xs font-semibold text-slate-300">Regime Probabilities at {currentStep.date}:</h4>
            {[
              { name: 'Risk-On', p: currentStep.pRiskOn, color: '#10b981' },
              { name: 'Late-Cycle', p: currentStep.pLateCycle, color: '#f59e0b' },
              { name: 'Transitional', p: currentStep.pTransitional, color: '#818cf8' },
              { name: 'Post-Shock', p: currentStep.pPostShock, color: '#06b6d4' },
              { name: 'Risk-Off', p: currentStep.pRiskOff, color: '#f43f5e' }
            ].map((reg, idx) => (
              <div key={idx} className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-300">{reg.name}</span>
                  <span className="text-white font-bold">{(reg.p * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{ width: `${reg.p * 100}%`, backgroundColor: reg.color }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right (5 cols): Timeline Progression Summary */}
        <div className="lg:col-span-5 glass-panel p-5 rounded-2xl border border-white/5 bg-slate-900/60 space-y-4">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <TrendingDown className="w-4 h-4 text-rose-400" />
            <span>Crisis Sequence Trace</span>
          </h3>

          <div className="space-y-3 relative before:absolute before:inset-0 before:left-3 before:w-0.5 before:bg-slate-800">
            {crisis.timeline.map((step, idx) => {
              const isActive = idx === currentStepIdx;
              const isPast = idx < currentStepIdx;

              return (
                <div
                  key={idx}
                  onClick={() => {
                    setCurrentStepIdx(idx);
                    setIsPlaying(false);
                  }}
                  className={`relative flex items-start gap-3 p-2.5 rounded-xl cursor-pointer transition-all ${
                    isActive
                      ? 'bg-white/10 border border-white/20'
                      : 'hover:bg-white/5'
                  }`}
                >
                  <div
                    className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-mono font-bold shrink-0 z-10 transition-colors ${
                      isActive
                        ? 'bg-rose-500 text-white shadow-md shadow-rose-500/30'
                        : isPast
                        ? 'bg-slate-700 text-slate-300'
                        : 'bg-slate-800 text-slate-500'
                    }`}
                  >
                    {idx + 1}
                  </div>
                  <div className="text-xs">
                    <div className="flex items-center gap-2 font-mono">
                      <span className={isActive ? 'text-white font-bold' : 'text-slate-300'}>
                        {step.date}
                      </span>
                      <span className="text-[11px] text-slate-500">
                        Nifty: {step.nifty}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400 mt-0.5 line-clamp-2">
                      {step.action}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
