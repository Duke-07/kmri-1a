import React, { useState } from 'react';
import { Cpu, ArrowRight, Code, Clock, CheckCircle2, ChevronRight, Layers } from './Icons.jsx';
import { PIPELINE_STAGES } from '../engine/data';

export default function PipelineArchitectureTab() {
  const [selectedStageId, setSelectedStageId] = useState(1);

  const selectedStage = PIPELINE_STAGES.find((s) => s.id === selectedStageId) || PIPELINE_STAGES[0];

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="glass-panel p-5 rounded-2xl border border-cyan-500/20 bg-gradient-to-r from-slate-900/90 via-slate-900/60 to-cyan-950/30">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-cyan-400 mb-1">
              <Cpu className="w-3.5 h-3.5" />
              <span>Full End-to-End Execution Graph</span>
              <span>•</span>
              <span>Total Runtime: ~13.8 Seconds</span>
            </div>
            <h2 className="text-2xl font-bold tracking-tight text-white">
              12-Stage Mathematical Pipeline Architecture
            </h2>
            <p className="text-xs text-slate-300 mt-1 max-w-2xl">
              Zero-loophole architecture orchestrated sequentially from Student-t synthetic market calibration to Bayesian variational inference, topological data analysis, foundation embeddings, and conformal prediction guarantees.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="px-3 py-1.5 rounded-lg bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 font-mono text-xs font-semibold">
              Pure Python & R Reconciliation
            </span>
          </div>
        </div>
      </div>

      {/* Stage Flow Nodes Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {PIPELINE_STAGES.map((stage) => {
          const isSelected = stage.id === selectedStageId;
          return (
            <button
              key={stage.id}
              onClick={() => setSelectedStageId(stage.id)}
              className={`p-3.5 rounded-xl text-left transition-all border ${
                isSelected
                  ? 'bg-indigo-600/20 border-indigo-500/40 shadow-sm shadow-indigo-500/10'
                  : 'bg-slate-900/60 border-white/5 hover:border-white/10 hover:bg-slate-900/90'
              }`}
            >
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="font-mono text-slate-500 text-[10px]">STAGE {stage.id}</span>
                <span className="font-mono text-indigo-400 text-[10px] flex items-center gap-1">
                  <Clock className="w-2.5 h-2.5" /> {stage.time}
                </span>
              </div>
              <div className="text-xs font-bold text-white line-clamp-1">{stage.title}</div>
            </button>
          );
        })}
      </div>

      {/* Selected Stage Deep-Dive Card */}
      <div className="glass-panel p-6 rounded-2xl border border-white/10 bg-slate-900/80 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-white/10 pb-4">
          <div>
            <div className="text-xs font-mono text-cyan-400">
              STAGE {selectedStage.id} OF 12
            </div>
            <h3 className="text-xl font-bold text-white mt-0.5">{selectedStage.title}</h3>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="px-2.5 py-1 rounded-md bg-white/5 text-slate-300">
              Execution: {selectedStage.time}
            </span>
            <span className="px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Status: Verified
            </span>
          </div>
        </div>

        {/* Mathematical Formulation */}
        <div className="space-y-2">
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wider font-mono">
            Mathematical Objective & Formula
          </span>
          <div className="p-3 rounded-xl bg-slate-950 border border-white/5 font-mono text-xs text-indigo-300 overflow-x-auto">
            <code>{selectedStage.formula}</code>
          </div>
        </div>

        {/* Description & Engineering Details */}
        <div className="space-y-2">
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wider font-mono">
            Algorithmic Specification & Fallbacks
          </span>
          <p className="text-xs text-slate-300 font-mono leading-relaxed bg-slate-950/40 p-4 rounded-xl border border-white/5">
            {selectedStage.desc}
          </p>
        </div>

        {/* Navigation between stages */}
        <div className="flex justify-between items-center pt-2">
          <button
            onClick={() => setSelectedStageId((prev) => Math.max(1, prev - 1))}
            disabled={selectedStageId === 1}
            className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 text-xs font-mono disabled:opacity-30 disabled:cursor-not-allowed"
          >
            ← Previous Stage
          </button>
          <span className="text-xs font-mono text-slate-500">
            Stage {selectedStageId} of 12
          </span>
          <button
            onClick={() => setSelectedStageId((prev) => Math.min(12, prev + 1))}
            disabled={selectedStageId === 12}
            className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-mono disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Next Stage →
          </button>
        </div>
      </div>
    </div>
  );
}
