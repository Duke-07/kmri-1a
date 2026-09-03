import React from 'react';
import { TrendingUp, ShieldAlert, BarChart3, CheckCircle2 } from './Icons.jsx';
import { BACKTEST_METRICS, CONFORMAL_CALIBRATION } from '../engine/data';

export default function MetricCards() {
  const cards = [
    {
      title: 'Information Ratio',
      value: `+${BACKTEST_METRICS.informationRatio}`,
      subtitle: 'Walk-Forward OOS · Simulated (18 Yrs)',
      badge: 'Alpha Positive',
      accent: 'stat-card--emerald',
      icon: TrendingUp,
      iconColor: 'text-emerald-600',
      iconBg: 'bg-emerald-50 border-emerald-200',
      badgeClass: 'bg-emerald-50 text-emerald-700 border border-emerald-200',
      valueColor: 'text-emerald-700',
    },
    {
      title: 'Conformal Marginal Coverage',
      value: `${(CONFORMAL_CALIBRATION.marginalCoverage.aps * 100).toFixed(1)}%`,
      subtitle: 'Target: 90.0% Marginal',
      badge: 'Exact Finite Sample',
      accent: 'stat-card--indigo',
      icon: CheckCircle2,
      iconColor: 'text-indigo-600',
      iconBg: 'bg-indigo-50 border-indigo-200',
      badgeClass: 'bg-indigo-50 text-indigo-700 border border-indigo-200',
      valueColor: 'text-indigo-700',
    },
    {
      title: 'Deflated Sharpe Ratio',
      value: `${BACKTEST_METRICS.deflatedSharpeRatio}`,
      subtitle: 'Bailey & López de Prado (2014)',
      badge: 'Passed p < 0.05',
      accent: 'stat-card--cyan',
      icon: BarChart3,
      iconColor: 'text-cyan-600',
      iconBg: 'bg-cyan-50 border-cyan-200',
      badgeClass: 'bg-cyan-50 text-cyan-700 border border-cyan-200',
      valueColor: 'text-cyan-700',
    },
    {
      title: 'Max Drawdown Delta',
      value: BACKTEST_METRICS.ddReduction,
      subtitle: 'Over Benchmark Buy & Hold',
      badge: '-32.1% vs -54.4%',
      accent: 'stat-card--amber',
      icon: ShieldAlert,
      iconColor: 'text-amber-600',
      iconBg: 'bg-amber-50 border-amber-200',
      badgeClass: 'bg-amber-50 text-amber-700 border border-amber-200',
      valueColor: 'text-amber-700',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {cards.map((card, idx) => {
        const Icon = card.icon;
        return (
          <div key={idx} className={`stat-card p-4 ${card.accent}`}>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide">{card.title}</span>
              <div className={`p-2 rounded-lg border ${card.iconBg}`}>
                <Icon className={`w-4 h-4 ${card.iconColor}`} />
              </div>
            </div>
            <div className="flex items-baseline gap-2 flex-wrap">
              <span className={`text-2xl font-bold tracking-tight font-mono tabular-nums ${card.valueColor}`}>
                {card.value}
              </span>
              <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono font-semibold ${card.badgeClass}`}>
                {card.badge}
              </span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1.5 font-mono">{card.subtitle}</p>
          </div>
        );
      })}
    </div>
  );
}
