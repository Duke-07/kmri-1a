import React, { useState, useMemo } from 'react';
import { Sliders, RefreshCw, Zap, Shield, AlertTriangle, Layers, ArrowUpRight, ArrowDownRight, Compass } from './Icons.jsx';
import { REGIMES, INITIAL_LIVE_STATE } from '../engine/data';

export default function LiveTerminalTab() {
  // Interactive feature perturbations
  const [vix, setVix] = useState(13.45);
  const [fiiFlow, setFiiFlow] = useState(1842); // in Crores
  const [mcclellan, setMcclellan] = useState(42.5);
  const [tdaNorm, setTdaNorm] = useState(1.42);

  // Dynamic Bayesian Regime Calculation based on inputs
  const { probs, dominant, entropy, uncertainty, tilt, predictionSet } = useMemo(() => {
    // Normalised scores
    // Risk-On features: low vix (<15), high fii (>1000), positive mcclellan (>20), low tda (<2.0)
    // Risk-Off features: high vix (>28), heavy fii selling (<-3000), negative mcclellan (<-40), high tda (>3.5)
    // Post-Shock features: high vix falling, oversold mcclellan (<-70), positive fii rebound
    // Late-Cycle features: moderate vix (15-22), low positive fii, declining breadth (mcclellan <0)
    // Transitional features: volatile vix, mixed flows, elevated tda

    let wOn = Math.max(0.01, (35 - vix) * 1.5 + (fiiFlow / 800) + (mcclellan / 25) - (tdaNorm - 1.5) * 4);
    let wLate = Math.max(0.01, (vix > 14 && vix < 24 ? 20 : 5) - (mcclellan / 30) + 10);
    let wTrans = Math.max(0.01, (tdaNorm > 2.0 ? 18 : 6) + Math.abs(mcclellan) / 20 + 8);
    let wPost = Math.max(0.01, (vix > 25 && mcclellan < -20 ? 25 : 2) + (fiiFlow > 0 ? 10 : 0));
    let wOff = Math.max(0.01, (vix - 16) * 1.8 - (fiiFlow / 600) - (mcclellan / 20) + (tdaNorm - 1.5) * 6);

    // Exponentiate (Softmax) for proper Bayesian probability simplex
    const exps = [wOn, wLate, wTrans, wPost, wOff].map((w) => Math.exp(Math.min(50, Math.max(-50, w / 8))));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    const rawProbs = exps.map((e) => e / sumExp);

    const calculatedProbs = {
      risk_on: rawProbs[0],
      late_cycle: rawProbs[1],
      transitional: rawProbs[2],
      post_shock: rawProbs[3],
      risk_off: rawProbs[4]
    };

    // Shannon Entropy: H(p) = -sum p_i ln p_i
    let totalEntropy = 0;
    Object.values(calculatedProbs).forEach((p) => {
      if (p > 1e-6) totalEntropy -= p * Math.log(p);
    });

    // Find dominant
    const entries = [
      { id: 'risk_on', name: 'Risk-On', prob: calculatedProbs.risk_on },
      { id: 'late_cycle', name: 'Late-Cycle', prob: calculatedProbs.late_cycle },
      { id: 'transitional', name: 'Transitional', prob: calculatedProbs.transitional },
      { id: 'post_shock', name: 'Post-Shock', prob: calculatedProbs.post_shock },
      { id: 'risk_off', name: 'Risk-Off', prob: calculatedProbs.risk_off }
    ].sort((a, b) => b.prob - a.prob);

    const dom = entries[0];

    // Conformal prediction set @ 90%
    let cum = 0;
    const pSet = [];
    for (const r of entries) {
      pSet.push(r.name);
      cum += r.prob;
      if (cum >= 0.90) break;
    }

    // Tactical Kelly Tilt (bounded [-5%, +5%])
    let computedTilt = '+0.0%';
    if (dom.id === 'risk_on') computedTilt = `+${(dom.prob * 5.0).toFixed(1)}% Beta`;
    else if (dom.id === 'risk_off') computedTilt = `-${(dom.prob * 5.0).toFixed(1)}% Beta`;
    else if (dom.id === 'post_shock') computedTilt = `+${(dom.prob * 3.5).toFixed(1)}% Beta`;
    else if (dom.id === 'late_cycle') computedTilt = 'Neutral (Quality Tilt)';
    else computedTilt = '-2.5% Cash Buffer';

    const aleatoric = Math.min(92, Math.max(45, 100 - totalEntropy * 35));
    const epistemic = 100 - aleatoric;

    return {
      probs: calculatedProbs,
      dominant: dom,
      entropy: totalEntropy.toFixed(3),
      uncertainty: { aleatoric: aleatoric.toFixed(1), epistemic: epistemic.toFixed(1) },
      tilt: computedTilt,
      predictionSet: pSet
    };
  }, [vix, fiiFlow, mcclellan, tdaNorm]);

  const handleReset = () => {
    setVix(13.45);
    setFiiFlow(1842);
    setMcclellan(42.5);
    setTdaNorm(1.42);
  };

  return (
    <div className="space-y-6">
      {/* Top Banner: Market Status & Dominant Regime Callout */}
      <div className="glass-panel p-5 rounded-2xl border border-white/10 bg-gradient-to-r from-slate-900/90 via-slate-900/70 to-indigo-950/40">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
                Active Regime State
              </span>
              <span className="text-slate-600">•</span>
              <span className="text-xs font-mono text-cyan-400">NIFTY 50: {INITIAL_LIVE_STATE.niftyCurrent.toLocaleString('en-IN')}</span>
              <span className="text-xs font-mono text-emerald-400">{INITIAL_LIVE_STATE.niftyChange}</span>
            </div>
            <div className="flex items-baseline gap-3">
              <h2 className="text-3xl font-extrabold tracking-tight text-white flex items-center gap-2">
                <span style={{ color: REGIMES.find(r => r.id === dominant.id)?.color }}>
                  {dominant.name}
                </span>
                <span className="text-base font-normal font-mono text-slate-400">
                  ({(dominant.prob * 100).toFixed(1)}% Probability)
                </span>
              </h2>
            </div>
            <p className="text-xs text-slate-300 mt-1 max-w-2xl">
              {REGIMES.find(r => r.id === dominant.id)?.description}
            </p>
          </div>

          {/* Allocation Tilt & SEBI Risk Scale */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="px-4 py-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20">
              <div className="text-[10px] uppercase font-mono tracking-wider text-indigo-300">
                Tactical Overlay Tilt
              </div>
              <div className="text-sm font-bold font-mono text-indigo-400">
                {tilt}
              </div>
            </div>

            <div className="px-4 py-2 rounded-xl bg-white/5 border border-white/10">
              <div className="text-[10px] uppercase font-mono tracking-wider text-slate-400">
                Conformal Set Ĉ(X)
              </div>
              <div className="text-sm font-bold font-mono text-white flex items-center gap-1">
                {predictionSet.map((item, idx) => (
                  <span key={idx} className="px-1.5 py-0.5 rounded bg-white/10 text-xs">
                    {item}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Grid: Probabilities vs Telemetry */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Probability Simplex (7 cols) */}
        <div className="lg:col-span-7 glass-panel p-5 rounded-2xl border border-white/5 bg-slate-900/60 space-y-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-400" />
              <h3 className="text-sm font-semibold text-white">
                5-State Bayesian Probability Simplex
              </h3>
            </div>
            <span className="text-xs font-mono text-slate-400">
              Shannon Entropy: {entropy} nats
            </span>
          </div>

          {/* Bars for 5 Regimes */}
          <div className="space-y-4">
            {REGIMES.map((regime) => {
              const p = probs[regime.id] || 0;
              const pct = (p * 100).toFixed(1);
              const isDom = dominant.id === regime.id;

              return (
                <div key={regime.id} className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <span
                        className="w-2.5 h-2.5 rounded-full"
                        style={{ backgroundColor: regime.color }}
                      ></span>
                      <span className={`font-medium ${isDom ? 'text-white font-bold' : 'text-slate-300'}`}>
                        {regime.name}
                      </span>
                      <span className="text-[10px] font-mono text-slate-500">
                        ({regime.code})
                      </span>
                    </div>
                    <div className="flex items-center gap-2 font-mono">
                      <span className="text-slate-400 text-[11px]">{regime.allocationTilt}</span>
                      <span className={`font-bold tabular-nums ${isDom ? 'text-white' : 'text-slate-300'}`}>
                        {pct}%
                      </span>
                    </div>
                  </div>

                  {/* Progress Bar */}
                  <div className="w-full h-2.5 rounded-full bg-slate-800/80 overflow-hidden relative">
                    <div
                      className="h-full rounded-full transition-all duration-300 ease-out"
                      style={{
                        width: `${Math.max(1, p * 100)}%`,
                        backgroundColor: regime.color,
                        boxShadow: isDom ? `0 0 12px ${regime.color}` : 'none'
                      }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Uncertainty Budget: Aleatoric vs Epistemic */}
          <div className="pt-4 border-t border-white/5">
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="text-slate-300 font-medium">Uncertainty Budget Decomposition</span>
              <span className="text-slate-400 font-mono text-[11px]">
                Aleatoric: {uncertainty.aleatoric}% | Epistemic: {uncertainty.epistemic}%
              </span>
            </div>
            <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden flex">
              <div
                className="bg-indigo-500 h-full transition-all duration-300"
                style={{ width: `${uncertainty.aleatoric}%` }}
                title="Aleatoric (Market stochasticity)"
              ></div>
              <div
                className="bg-cyan-400 h-full transition-all duration-300"
                style={{ width: `${uncertainty.epistemic}%` }}
                title="Epistemic (Model ignorance)"
              ></div>
            </div>
            <p className="text-[11px] text-slate-400 mt-2 font-mono">
              {INITIAL_LIVE_STATE.uncertaintyBreakdown.interpretation}
            </p>
          </div>
        </div>

        {/* Right Column: Live Telemetry & Interactive Feature Perturbations (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Interactive Feature Sliders */}
          <div className="glass-panel p-5 rounded-2xl border border-white/5 bg-slate-900/60 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-semibold text-white">Feature Perturbation Sandbox</h3>
              </div>
              <button
                onClick={handleReset}
                className="text-[11px] font-mono text-slate-400 hover:text-white flex items-center gap-1 transition-colors"
                title="Reset to current market telemetry"
              >
                <RefreshCw className="w-3 h-3" /> Reset
              </button>
            </div>
            <p className="text-xs text-slate-400">
              Modify market signals in real time to observe the Bayesian ensemble's posterior response:
            </p>

            {/* Slider 1: India VIX */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-slate-300">India VIX</span>
                <span className="text-cyan-400 font-bold">{vix.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="9"
                max="65"
                step="0.5"
                value={vix}
                onChange={(e) => setVix(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                <span>9.0 (Complacent)</span>
                <span>28.0 (Crisis Barrier)</span>
                <span>65.0 (Extreme)</span>
              </div>
            </div>

            {/* Slider 2: FII Net Flow */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-slate-300">FII Net Flow (₹ Cr)</span>
                <span className={`font-bold ${fiiFlow >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {fiiFlow >= 0 ? `+₹${fiiFlow}` : `-₹${Math.abs(fiiFlow)}`}
                </span>
              </div>
              <input
                type="range"
                min="-8000"
                max="5000"
                step="250"
                value={fiiFlow}
                onChange={(e) => setFiiFlow(parseFloat(e.target.value))}
                className="w-full accent-indigo-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                <span>-₹8,000 Cr</span>
                <span>0</span>
                <span>+₹5,000 Cr</span>
              </div>
            </div>

            {/* Slider 3: McClellan Oscillator */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-slate-300">McClellan Market Breadth</span>
                <span className={`font-bold ${mcclellan >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {mcclellan > 0 ? `+${mcclellan.toFixed(1)}` : mcclellan.toFixed(1)}
                </span>
              </div>
              <input
                type="range"
                min="-100"
                max="100"
                step="2"
                value={mcclellan}
                onChange={(e) => setMcclellan(parseFloat(e.target.value))}
                className="w-full accent-emerald-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                <span>-100 (Oversold)</span>
                <span>0 (Neutral)</span>
                <span>+100 (Overbought)</span>
              </div>
            </div>

            {/* Slider 4: TDA Persistence Norm */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-slate-300">TDA Persistence Landscape L2 Norm</span>
                <span className="text-amber-400 font-bold">{tdaNorm.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="6.0"
                step="0.1"
                value={tdaNorm}
                onChange={(e) => setTdaNorm(parseFloat(e.target.value))}
                className="w-full accent-amber-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                <span>0.5 (Regular)</span>
                <span>3.0 (Early Warning)</span>
                <span>6.0 (Crash Geometry)</span>
              </div>
            </div>
          </div>

          {/* Quick Engine Telemetry Specs */}
          <div className="glass-panel p-4 rounded-xl border border-white/5 bg-slate-900/60 grid grid-cols-2 gap-3 text-xs font-mono">
            <div>
              <div className="text-slate-500 text-[10px]">PARKINSON VOL</div>
              <div className="text-white font-semibold">{INITIAL_LIVE_STATE.parkinsonVol}</div>
            </div>
            <div>
              <div className="text-slate-500 text-[10px]">PSI POPULATION DRIFT</div>
              <div className="text-emerald-400 font-semibold">{INITIAL_LIVE_STATE.psiDrift}</div>
            </div>
            <div>
              <div className="text-slate-500 text-[10px]">DII INSTITUTIONAL NET</div>
              <div className="text-white font-semibold">{INITIAL_LIVE_STATE.diiNetFlow}</div>
            </div>
            <div>
              <div className="text-slate-500 text-[10px]">RECOMMENDED RISK-O-METER</div>
              <div className="text-amber-400 font-semibold">{REGIMES.find(r => r.id === dominant.id)?.riskOMeter}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
