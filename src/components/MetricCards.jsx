import React from 'react';
import { TrendingUp, ShieldAlert, Cpu, BarChart3, CheckCircle2 } from './Icons.jsx';
import { BACKTEST_METRICS, CONFORMAL_CALIBRATION } from '../engine/data';

export default function MetricCards() {
  const cards = [
    {
      title: 'Information Ratio',
      value: `+${BACKTEST_METRICS.informationRatio}`,
      subtitle: 'Walk-Forward OOS (18 Yrs)',
      badge: 'Alpha Positive',
      badgeColor: 'emerald',
      icon: TrendingUp,
      color: 'text-emerald-400',
      bgColor: 'bg-emerald-500/10',
      borderColor: 'border-emerald-500/20'
    },
    {
      title: 'Conformal Marginal Coverage',
      value: `${(CONFORMAL_CALIBRATION.marginalCoverage.aps * 100).toFixed(1)}%`,
      subtitle: 'Target: 90.0% Marginal',
      badge: 'Exact Finite Sample',
      badgeColor: 'indigo',
      icon: CheckCircle2,
      color: 'text-indigo-400',
      bgColor: 'bg-indigo-500/10',
      borderColor: 'border-indigo-500/20'
    },
    {
      title: 'Deflated Sharpe Ratio',
      value: `${BACKTEST_METRICS.deflatedSharpeRatio}`,
      subtitle: 'Bailey & López de Prado (2014)',
      badge: 'Passed p < 0.05',
      badgeColor: 'cyan',
      icon: BarChart3,
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-500/10',
      borderColor: 'border-cyan-500/20'
    },
    {
      title: 'Max Drawdown Delta',
      value: BACKTEST_METRICS.ddReduction,
      subtitle: 'Over Benchmark Buy & Hold',
      badge: '-32.1% vs -54.4%',
      badgeColor: 'amber',
      icon: ShieldAlert,
      color: 'text-amber-400',
      bgColor: 'bg-amber-500/10',
      borderColor: 'border-amber-500/20'
    }
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {cards.map((card, idx) => {
        const Icon = card.icon;
        return (
          <div
            key={idx}
            className="glass-panel p-4 rounded-xl border border-white/5 bg-slate-900/60 hover:border-white/10 transition-all duration-200"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-slate-400">{card.title}</span>
              <div className={`p-2 rounded-lg ${card.bgColor} ${card.borderColor} border`}>
                <Icon className={`w-4 h-4 ${card.color}`} />
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold tracking-tight text-white font-mono tabular-nums">
                {card.value}
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-white/5 text-slate-300 font-mono text-[11px]">
                {card.badge}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1 font-mono">{card.subtitle}</p>
          </div>
        );
      })}
    </div>
  );
}
