import React from 'react';
import { TrendingUp, ShieldAlert, BarChart3, CheckCircle2 } from './Icons.jsx';
import { BACKTEST_METRICS, CONFORMAL_CALIBRATION } from '../engine/data';

export default function MetricCards() {
  const cards = [
    {
      title: 'Information Ratio',
      value: `+${BACKTEST_METRICS.informationRatio}`,
      subtitle: 'Walk-Forward OOS (18 Yrs)',
      badge: 'Alpha Positive',
      icon: TrendingUp,
      iconColor: 'text-emerald-600 dark:text-emerald-400',
      bgColor: 'bg-emerald-50 dark:bg-emerald-500/10',
      borderColor: 'border-emerald-200 dark:border-emerald-500/20',
      badgeClass: 'bg-emerald-50 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800'
    },
    {
      title: 'Conformal Marginal Coverage',
      value: `${(CONFORMAL_CALIBRATION.marginalCoverage.aps * 100).toFixed(1)}%`,
      subtitle: 'Target: 90.0% Marginal',
      badge: 'Exact Finite Sample',
      icon: CheckCircle2,
      iconColor: 'text-indigo-600 dark:text-indigo-400',
      bgColor: 'bg-indigo-50 dark:bg-indigo-500/10',
      borderColor: 'border-indigo-200 dark:border-indigo-500/20',
      badgeClass: 'bg-indigo-50 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-400 border border-indigo-200 dark:border-indigo-800'
    },
    {
      title: 'Deflated Sharpe Ratio',
      value: `${BACKTEST_METRICS.deflatedSharpeRatio}`,
      subtitle: 'Bailey & López de Prado (2014)',
      badge: 'Passed p < 0.05',
      icon: BarChart3,
      iconColor: 'text-cyan-600 dark:text-cyan-400',
      bgColor: 'bg-cyan-50 dark:bg-cyan-500/10',
      borderColor: 'border-cyan-200 dark:border-cyan-500/20',
      badgeClass: 'bg-cyan-50 dark:bg-cyan-950 text-cyan-700 dark:text-cyan-400 border border-cyan-200 dark:border-cyan-800'
    },
    {
      title: 'Max Drawdown Delta',
      value: BACKTEST_METRICS.ddReduction,
      subtitle: 'Over Benchmark Buy & Hold',
      badge: '-32.1% vs -54.4%',
      icon: ShieldAlert,
      iconColor: 'text-amber-600 dark:text-amber-400',
      bgColor: 'bg-amber-50 dark:bg-amber-500/10',
      borderColor: 'border-amber-200 dark:border-amber-500/20',
      badgeClass: 'bg-amber-50 dark:bg-amber-950 text-amber-700 dark:text-amber-400 border border-amber-200 dark:border-amber-800'
    }
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {cards.map((card, idx) => {
        const Icon = card.icon;
        return (
          <div
            key={idx}
            className="quant-card p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">{card.title}</span>
              <div className={`p-2 rounded-lg ${card.bgColor} ${card.borderColor} border`}>
                <Icon className={`w-4 h-4 ${card.iconColor}`} />
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white font-mono tabular-nums">
                {card.value}
              </span>
              <span className={`text-[11px] px-2 py-0.5 rounded-full font-mono font-medium ${card.badgeClass}`}>
                {card.badge}
              </span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 font-mono">{card.subtitle}</p>
          </div>
        );
      })}
    </div>
  );
}
