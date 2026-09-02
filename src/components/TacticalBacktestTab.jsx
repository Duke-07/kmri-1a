import React, { useState } from 'react';
import { TrendingUp, ShieldAlert, Sliders, DollarSign, BarChart2, PieChart, Activity } from './Icons.jsx';
import { BACKTEST_METRICS } from '../engine/data';

export default function TacticalBacktestTab() {
  const [kellyFraction, setKellyFraction] = useState(0.5); // Half-Kelly default

  // Cumulative equity curve points across 18 years (2007 to 2024)
  const equityPoints = [
    { year: '2007', bench: 100, strat: 100, note: 'Base Index = 100' },
    { year: '2008', bench: 48, strat: 72, note: 'GFC Crash: +24pp alpha via Risk-Off tilt' },
    { year: '2010', bench: 124, strat: 146, note: 'Post-Shock recovery capture' },
    { year: '2013', bench: 118, strat: 154, note: 'Taper Tantrum drawdown mitigated' },
    { year: '2016', bench: 162, strat: 218, note: 'Demonetisation buffer' },
    { year: '2018', bench: 210, strat: 278, note: 'IL&FS Midcap divergence de-risking' },
    { year: '2020', bench: 232, strat: 334, note: 'COVID: TDA 18-day lead warning' },
    { year: '2021', bench: 345, strat: 472, note: 'Massive Risk-On momentum capture' },
    { year: '2022', bench: 358, strat: 498, note: 'Global rate hike cycle defense' },
    { year: '2024', bench: 485, strat: 672, note: 'NIFTY 24k+ ATH: 6.72x total return' }
  ];

  const maxStrat = 672;

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="glass-panel p-5 rounded-2xl border border-emerald-500/20 bg-gradient-to-r from-slate-900/90 via-slate-900/60 to-emerald-950/30">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-emerald-400 mb-1">
              <span>Purged & Embargoed Walk-Forward (18 Years)</span>
              <span>•</span>
              <span>Kelly Criterion Overlay (Bounded ±5%)</span>
            </div>
            <h2 className="text-2xl font-bold tracking-tight text-white">
              Tactical Allocation & Walk-Forward Backtest
            </h2>
            <p className="text-xs text-slate-300 mt-1 max-w-2xl">
              Strict walk-forward evaluation using combinatorial purged cross-validation. The engine tilts portfolio equity beta based on regime conviction, delivering <span className="text-emerald-400 font-bold font-mono">IR = +0.6142</span> with a 22.3 percentage-point drawdown reduction.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="px-3.5 py-2 rounded-xl bg-slate-900/80 border border-white/10 text-right font-mono text-xs">
              <div className="text-[10px] text-slate-400">INFORMATION RATIO</div>
              <div className="text-emerald-400 font-bold text-base">+{BACKTEST_METRICS.informationRatio}</div>
            </div>
            <div className="px-3.5 py-2 rounded-xl bg-slate-900/80 border border-white/10 text-right font-mono text-xs">
              <div className="text-[10px] text-slate-400">TRACKING ERROR (ANN.)</div>
              <div className="text-cyan-400 font-bold text-base">{BACKTEST_METRICS.trackingErrorAnn}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Kelly Fraction Sizing Slider */}
      <div className="glass-panel p-5 rounded-2xl border border-white/5 bg-slate-900/60">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-2">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-emerald-400" />
            <span className="text-sm font-semibold text-white">Kelly Leverage Multiplier (λ)</span>
          </div>
          <div className="font-mono text-xs text-slate-300">
            Current: <strong className="text-emerald-400 font-bold">{kellyFraction === 0.5 ? 'Half-Kelly (λ = 0.50)' : `λ = ${kellyFraction.toFixed(2)}`}</strong>
            <span className="text-slate-500 ml-2">Bounded Max Overlay: ±5.0% Beta</span>
          </div>
        </div>
        <input
          type="range"
          min="0.1"
          max="1.0"
          step="0.05"
          value={kellyFraction}
          onChange={(e) => setKellyFraction(parseFloat(e.target.value))}
          className="w-full accent-emerald-500 cursor-pointer h-2 bg-slate-800 rounded-lg"
        />
        <div className="flex justify-between text-[11px] text-slate-500 font-mono mt-1">
          <span>0.10 (Conservative)</span>
          <span>0.50 (Recommended Half-Kelly)</span>
          <span>1.00 (Full-Kelly Growth)</span>
        </div>
      </div>

      {/* Equity Curve Visualizer */}
      <div className="glass-panel p-5 rounded-2xl border border-white/5 bg-slate-900/60 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-emerald-400" />
              <span>Cumulative Growth of ₹100 Invested (2007 – 2024)</span>
            </h3>
            <p className="text-xs text-slate-400">
              Comparing Bayesian Regime Tactical Overlay vs Buy & Hold Benchmark (NIFTY 50 TRI)
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono">
            <span className="flex items-center gap-1.5 text-emerald-400">
              <span className="w-3 h-0.5 bg-emerald-400 inline-block"></span>
              Regime Overlay: ₹672 (+572%)
            </span>
            <span className="flex items-center gap-1.5 text-slate-400">
              <span className="w-3 h-0.5 bg-slate-400 inline-block"></span>
              Benchmark: ₹485 (+385%)
            </span>
          </div>
        </div>

        {/* Bar/Progress Visualisation of Growth */}
        <div className="space-y-3 pt-2">
          {equityPoints.map((pt, idx) => (
            <div key={idx} className="space-y-1 text-xs font-mono">
              <div className="flex justify-between text-slate-400 text-[11px]">
                <span className="text-white font-medium">{pt.year}</span>
                <span className="text-slate-500 hidden sm:inline">{pt.note}</span>
                <span>
                  Strategy: <strong className="text-emerald-400">₹{pt.strat}</strong> | Bench: ₹{pt.bench}
                </span>
              </div>
              <div className="w-full h-2 rounded-full bg-slate-800/80 overflow-hidden relative flex">
                <div
                  className="bg-emerald-500 h-full rounded-full transition-all duration-300"
                  style={{ width: `${(pt.strat / maxStrat) * 100}%` }}
                ></div>
                <div
                  className="bg-slate-600 h-full opacity-60 rounded-full"
                  style={{ width: `${(pt.bench / maxStrat) * 100}%`, position: 'absolute', top: 0, left: 0 }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Monte Carlo & Risk Distribution Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs font-mono">
        <div className="glass-panel p-4 rounded-xl border border-white/5 bg-slate-900/60">
          <div className="text-slate-400 text-[10px] uppercase">1-YEAR MC MEAN RETURN</div>
          <div className="text-emerald-400 font-bold text-lg mt-1">{BACKTEST_METRICS.monteCarlo.meanReturn}</div>
          <div className="text-slate-500 text-[11px] mt-0.5">5,000 Regime-Conditioned Paths</div>
        </div>
        <div className="glass-panel p-4 rounded-xl border border-white/5 bg-slate-900/60">
          <div className="text-slate-400 text-[10px] uppercase">95% VALUE-AT-RISK (1-YR)</div>
          <div className="text-rose-400 font-bold text-lg mt-1">{BACKTEST_METRICS.monteCarlo.var95}</div>
          <div className="text-slate-500 text-[11px] mt-0.5">CVaR 95%: {BACKTEST_METRICS.monteCarlo.cvar95}</div>
        </div>
        <div className="glass-panel p-4 rounded-xl border border-white/5 bg-slate-900/60">
          <div className="text-slate-400 text-[10px] uppercase">MAX DRAWDOWN COMPARISON</div>
          <div className="text-amber-400 font-bold text-lg mt-1">{BACKTEST_METRICS.maxDrawdownStrategy}</div>
          <div className="text-slate-500 text-[11px] mt-0.5">vs {BACKTEST_METRICS.maxDrawdownBenchmark} Benchmark</div>
        </div>
        <div className="glass-panel p-4 rounded-xl border border-white/5 bg-slate-900/60">
          <div className="text-slate-400 text-[10px] uppercase">DEFLATED SHARPE RATIO (DSR)</div>
          <div className="text-cyan-400 font-bold text-lg mt-1">{BACKTEST_METRICS.deflatedSharpeRatio}</div>
          <div className="text-slate-500 text-[11px] mt-0.5">Bailey & López de Prado (2014)</div>
        </div>
      </div>
    </div>
  );
}
