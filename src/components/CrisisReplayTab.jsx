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
              className={`px-4 py-2 rounded-xl text-xs font-semibold whitespace-nowrap transition-all flex items-center gap-2 border ${
                isSelected
                  ? 'bg-rose-50 text-rose-700 border-rose-300 shadow-sm'
                  : 'bg-white text-slate-700 border-slate-200 hover:border-slate-300 hover:text-slate-900'
              }`}
            >
              <AlertOctagon className="w-3.5 h-3.5 text-rose-600" />
              <span>{c.name}</span>
            </button>
          );
        })}
      </div>

      {/* Scenario Hero Card */}
      <div className="hero-banner hero-banner--rose p-6">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-rose-700 mb-1 font-semibold">
              <span>{crisis.period}</span>
              <span>•</span>
              <span>BOCPD Changepoint: {crisis.bocpdConfidence}</span>
              <span>•</span>
              <span>Early Warning Lead: {crisis.leadDays}</span>
            </div>
            <h2 className="text-2xl font-extrabold tracking-tight text-slate-900">{crisis.name}</h2>
            <p className="text-xs text-slate-600 mt-1">{crisis.headline}</p>
          </div>

          <div className="flex items-center gap-3">
            <div className="px-4 py-2 rounded-xl bg-white border border-slate-200 text-right font-mono text-xs shadow-sm">
              <div className="text-[10px] text-slate-400 font-semibold">PEAK INDIA VIX</div>
              <div className="text-rose-600 font-bold text-sm">{crisis.keyFeatures.vix}</div>
            </div>
            <div className="px-4 py-2 rounded-xl bg-white border border-slate-200 text-right font-mono text-xs shadow-sm">
              <div className="text-[10px] text-slate-400 font-semibold">MAX DRAWDOWN</div>
              <div className="text-rose-600 font-bold text-sm">{crisis.keyFeatures.drawdown}</div>
            </div>
            <div className="px-4 py-2 rounded-xl bg-white border border-slate-200 text-right font-mono text-xs shadow-sm">
              <div className="text-[10px] text-slate-400 font-semibold">TDA L2 NORM</div>
              <div className="text-amber-600 font-bold text-sm">{crisis.keyFeatures.tdaNorm}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Interactive Playback Controller */}
      <div className="quant-card p-4 rounded-xl border border-slate-200 bg-white shadow-sm flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="btn-primary"
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{isPlaying ? 'Pause Timeline' : 'Play Timeline'}</span>
          </button>

          <button
            onClick={() => {
              setCurrentStepIdx(0);
              setIsPlaying(false);
            }}
            className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs transition-colors border border-slate-200"
            title="Reset replay to beginning"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          <span className="text-xs font-mono text-slate-500">
            Step {currentStepIdx + 1} of {crisis.timeline.length}
          </span>
        </div>

        {/* Scrubber step buttons */}
        <div className="flex items-center gap-1.5 overflow-x-auto max-w-full">
          {crisis.timeline.map((step, idx) => (
            <button
              key={idx}
              onClick={() => {
                setCurrentStepIdx(idx);
                setIsPlaying(false);
              }}
              className={`px-3 py-1 rounded-md text-[11px] font-mono transition-all border ${
                currentStepIdx === idx
                  ? 'bg-rose-600 text-white font-bold border-rose-600 shadow-sm'
                  : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100 hover:text-slate-900'
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
        <div className="lg:col-span-7 quant-card p-5 rounded-2xl border border-slate-200 bg-white shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div>
              <span className="text-[10px] uppercase font-mono tracking-wider text-slate-400 font-semibold">
                REPLAY DATE SNAPSHOT
              </span>
              <h3 className="text-xl font-bold font-mono text-slate-900 flex items-center gap-2">
                {currentStep.date}
                <span className="text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-700 font-normal border border-slate-200">
                  NIFTY: {currentStep.nifty.toLocaleString('en-IN')}
                </span>
              </h3>
            </div>
            <div className="text-right">
              <span className="text-[10px] uppercase font-mono tracking-wider text-slate-400 font-semibold">
                RISK-OFF POSTERIOR
              </span>
              <div className="text-xl font-bold font-mono text-rose-600">
                {(currentStep.pRiskOff * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Engine Action Card */}
          <div className="p-4 rounded-xl bg-indigo-50 border border-indigo-200 text-xs">
            <div className="flex items-center gap-1.5 text-indigo-700 font-bold mb-1">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Bayesian Engine Autonomous Response</span>
            </div>
            <p className="text-slate-800 font-mono text-[13px] leading-relaxed">
              "{currentStep.action}"
            </p>
          </div>

          {/* Probability Breakdown for this date */}
          <div className="space-y-3 pt-2">
            <h4 className="text-xs font-bold text-slate-700">Regime Probabilities at {currentStep.date}:</h4>
            {[
              { name: 'Risk-On', p: currentStep.pRiskOn, color: '#059669' },
              { name: 'Late-Cycle', p: currentStep.pLateCycle, color: '#d97706' },
              { name: 'Transitional', p: currentStep.pTransitional, color: '#4f46e5' },
              { name: 'Post-Shock', p: currentStep.pPostShock, color: '#0891b2' },
              { name: 'Risk-Off', p: currentStep.pRiskOff, color: '#e11d48' }
            ].map((reg, idx) => (
              <div key={idx} className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-600 font-medium">{reg.name}</span>
                  <span className="text-slate-900 font-bold">{(reg.p * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-2.5 rounded-full bg-slate-100 overflow-hidden">
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
        <div className="lg:col-span-5 quant-card p-5 rounded-2xl border border-slate-200 bg-white shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <TrendingDown className="w-4 h-4 text-rose-600" />
            <span>Crisis Sequence Trace</span>
          </h3>

          <div className="space-y-2.5 relative before:absolute before:inset-0 before:left-3 before:w-0.5 before:bg-slate-200">
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
                  className={`relative flex items-start gap-3 p-3 rounded-xl cursor-pointer transition-all border ${
                    isActive
                      ? 'bg-rose-50/80 border-rose-200 shadow-sm'
                      : 'bg-white border-transparent hover:border-slate-200'
                  }`}
                >
                  <div
                    className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-mono font-bold shrink-0 z-10 transition-colors ${
                      isActive
                        ? 'bg-rose-600 text-white shadow-sm'
                        : isPast
                        ? 'bg-slate-200 text-slate-700'
                        : 'bg-slate-100 text-slate-400'
                    }`}
                  >
                    {idx + 1}
                  </div>
                  <div className="text-xs">
                    <div className="flex items-center gap-2 font-mono">
                      <span className={isActive ? 'text-slate-900 font-bold' : 'text-slate-600'}>
                        {step.date}
                      </span>
                      <span className="text-[11px] text-slate-400">
                        Nifty: {step.nifty}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-500 mt-0.5 line-clamp-2 leading-normal">
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
