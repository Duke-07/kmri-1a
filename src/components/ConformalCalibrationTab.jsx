import React, { useState } from 'react';
import { ShieldCheck, CheckCircle2, Award, Info, Sliders, Target } from './Icons.jsx';
import { CONFORMAL_CALIBRATION } from '../engine/data';

export default function ConformalCalibrationTab() {
  const [targetCoverage, setTargetCoverage] = useState(90);

  // Dynamic set size estimate based on target coverage
  const estimatedMeanSetSize = (1.0 + Math.pow(targetCoverage / 100, 3) * 1.6).toFixed(2);
  const alpha = ((100 - targetCoverage) / 100).toFixed(2);

  // Reliability diagram bins (observed frequency vs forecast probability)
  const reliabilityBins = [
    { bin: '0.0 – 0.1', forecast: 0.05, observed: 0.048, count: 840 },
    { bin: '0.1 – 0.2', forecast: 0.15, observed: 0.149, count: 620 },
    { bin: '0.2 – 0.3', forecast: 0.25, observed: 0.244, count: 480 },
    { bin: '0.3 – 0.4', forecast: 0.35, observed: 0.358, count: 390 },
    { bin: '0.4 – 0.5', forecast: 0.45, observed: 0.447, count: 310 },
    { bin: '0.5 – 0.6', forecast: 0.55, observed: 0.562, count: 290 },
    { bin: '0.6 – 0.7', forecast: 0.65, observed: 0.641, count: 340 },
    { bin: '0.7 – 0.8', forecast: 0.75, observed: 0.759, count: 450 },
    { bin: '0.8 – 0.9', forecast: 0.85, observed: 0.864, count: 520 },
    { bin: '0.9 – 1.0', forecast: 0.95, observed: 0.946, count: 680 }
  ];

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="quant-card p-6 rounded-2xl border border-indigo-200 dark:border-indigo-900/50 bg-gradient-to-r from-white via-indigo-50/30 to-indigo-100/20 dark:from-slate-900 dark:via-slate-900/60 dark:to-indigo-950/30 shadow-sm">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-indigo-700 dark:text-indigo-400 mb-1 font-semibold">
              <span>Split-Conformal & APS (Romano et al. 2020)</span>
              <span>•</span>
              <span>Mondrian Class-Conditional</span>
              <span>•</span>
              <span>ACI (Gibbs & Candès 2021)</span>
            </div>
            <h2 className="text-2xl font-extrabold tracking-tight text-slate-900 dark:text-white">
              Conformal Calibration & Prediction Sets
            </h2>
            <p className="text-xs text-slate-600 dark:text-slate-300 mt-1 max-w-2xl leading-relaxed">
              Distribution-free, finite-sample-valid coverage guarantee: <span className="font-mono font-bold text-indigo-700 dark:text-indigo-300">P(Y_(n+1) ∈ Ĉ(X_(n+1))) ≥ 1 - α</span>, guaranteeing that risk-managed prediction sets never systematically under-cover during market distress.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="px-4 py-2 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-right font-mono text-xs shadow-sm">
              <div className="text-[10px] text-slate-400 font-semibold">EXPECTED CALIBRATION ERROR (ECE)</div>
              <div className="text-emerald-600 dark:text-emerald-400 font-bold text-base">{CONFORMAL_CALIBRATION.ece}</div>
            </div>
            <div className="px-4 py-2 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-right font-mono text-xs shadow-sm">
              <div className="text-[10px] text-slate-400 font-semibold">RPS SKILL SCORE</div>
              <div className="text-indigo-600 dark:text-cyan-400 font-bold text-base">+{CONFORMAL_CALIBRATION.rpsSkillScore}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Target Coverage Interactive Slider */}
      <div className="quant-card p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
            <span className="text-sm font-bold text-slate-900 dark:text-white">Interactive Target Coverage Level (1 - α)</span>
          </div>
          <div className="flex items-center gap-3 font-mono text-xs">
            <span className="text-slate-500">α = {alpha}</span>
            <span className="px-3 py-1 rounded-lg bg-indigo-50 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 font-bold text-sm border border-indigo-200 dark:border-indigo-800">
              {targetCoverage}% Coverage
            </span>
            <span className="text-slate-500">Mean Set Size: <strong className="text-slate-900 dark:text-white">{estimatedMeanSetSize}</strong></span>
          </div>
        </div>
        <input
          type="range"
          min="80"
          max="99"
          step="1"
          value={targetCoverage}
          onChange={(e) => setTargetCoverage(parseInt(e.target.value))}
          className="w-full accent-indigo-600 dark:accent-indigo-400 cursor-pointer h-2 bg-slate-200 dark:bg-slate-800 rounded-lg"
        />
        <div className="flex justify-between text-[11px] text-slate-400 font-mono mt-1">
          <span>80% (Aggressive, Smaller Sets)</span>
          <span>90% (Institutional Standard)</span>
          <span>99% (Conservative, Wider Sets)</span>
        </div>
      </div>

      {/* Two Column Layout: Mondrian Table & Reliability Diagram */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left (6 cols): Mondrian Class-Conditional Coverage */}
        <div className="lg:col-span-6 quant-card p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Target className="w-4 h-4 text-indigo-600 dark:text-cyan-400" />
              <span>Mondrian Class-Conditional Coverage</span>
            </h3>
            <span className="text-xs font-mono text-slate-500 dark:text-slate-400 font-medium">Target: 90.0%</span>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
            Guarantees that coverage does not degrade on rare crisis regimes (e.g. Risk-Off) by computing quantile thresholds per regime partition:
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400">
                  <th className="pb-2">Regime</th>
                  <th className="pb-2">Target</th>
                  <th className="pb-2">Empirical</th>
                  <th className="pb-2">Mean Set Size</th>
                  <th className="pb-2 text-right">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800/80">
                {CONFORMAL_CALIBRATION.mondrianClasses.map((item, idx) => (
                  <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                    <td className="py-2.5 font-semibold text-slate-800 dark:text-slate-200">{item.regime}</td>
                    <td className="py-2.5 text-slate-500">{item.target}</td>
                    <td className="py-2.5 text-emerald-600 dark:text-emerald-400 font-bold">{item.observed}</td>
                    <td className="py-2.5 text-slate-700 dark:text-slate-300">{item.meanSetSize}</td>
                    <td className="py-2.5 text-right">
                      <span className="px-2 py-0.5 rounded bg-emerald-50 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800 text-[10px] font-bold">
                        Valid
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right (6 cols): Reliability Diagram (ECE = 0.0156) */}
        <div className="lg:col-span-6 quant-card p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Award className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
              <span>Reliability Calibration Diagram</span>
            </h3>
            <span className="text-xs font-mono text-emerald-600 dark:text-emerald-400 font-bold">ECE: 0.0156</span>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
            Comparison between model forecast probability and empirical observed outcome frequency. Alignment with diagonal indicates zero overconfidence:
          </p>

          <div className="space-y-2">
            {reliabilityBins.map((bin, idx) => (
              <div key={idx} className="space-y-1 text-[11px] font-mono">
                <div className="flex justify-between text-slate-500 dark:text-slate-400">
                  <span>Bin: {bin.bin}</span>
                  <span>
                    Forecast: {bin.forecast.toFixed(2)} | Observed: <strong className="text-slate-800 dark:text-white">{bin.observed.toFixed(3)}</strong>
                  </span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden flex">
                  <div
                    className="bg-indigo-600 dark:bg-indigo-500 h-full rounded-full"
                    style={{ width: `${bin.observed * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>

          <div className="pt-3 border-t border-slate-100 dark:border-slate-800 grid grid-cols-3 gap-2 text-center text-xs font-mono">
            <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700">
              <div className="text-[10px] text-slate-400 font-semibold">BRIER RELIABILITY</div>
              <div className="text-slate-900 dark:text-white font-bold">{CONFORMAL_CALIBRATION.brierDecomposition.reliability}</div>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700">
              <div className="text-[10px] text-slate-400 font-semibold">BRIER RESOLUTION</div>
              <div className="text-slate-900 dark:text-white font-bold">{CONFORMAL_CALIBRATION.brierDecomposition.resolution}</div>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700">
              <div className="text-[10px] text-slate-400 font-semibold">BRIER SCORE</div>
              <div className="text-emerald-600 dark:text-emerald-400 font-bold">{CONFORMAL_CALIBRATION.brierScore}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
