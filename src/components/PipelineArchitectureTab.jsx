import React, { useState } from 'react';
import { Cpu, ArrowRight, Code, Clock, CheckCircle2, ChevronRight, Layers } from './Icons.jsx';
import { PIPELINE_STAGES } from '../engine/data';

export default function PipelineArchitectureTab() {
  const [selectedStageId, setSelectedStageId] = useState(1);

  const selectedStage = PIPELINE_STAGES.find((s) => s.id === selectedStageId) || PIPELINE_STAGES[0];

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="quant-card p-6 rounded-2xl border border-cyan-200 dark:border-cyan-900/50 bg-gradient-to-r from-white via-cyan-50/30 to-cyan-100/20 dark:from-slate-900 dark:via-slate-900/60 dark:to-cyan-950/30 shadow-sm">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-cyan-700 dark:text-cyan-400 mb-1 font-semibold">
              <Cpu className="w-3.5 h-3.5" />
              <span>Full End-to-End Execution Graph</span>
              <span>•</span>
              <span>Total Runtime: ~13.8 Seconds</span>
            </div>
            <h2 className="text-2xl font-extrabold tracking-tight text-slate-900">
              14-Stage Mathematical Pipeline Architecture
            </h2>
            <p className="text-xs text-slate-600 mt-1 max-w-2xl leading-relaxed">
              Rigorous sequential workflow from Student-t synthetic market calibration to Bayesian variational inference, topological data analysis, foundation embeddings, crisis replay harnesses, and conformal prediction guarantees.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="px-3.5 py-1.5 rounded-lg bg-cyan-50 text-cyan-800 border border-cyan-200 font-mono text-xs font-bold">
              Python 3.10+ & R 4.3+ Reconciled
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
                  ? 'bg-indigo-50 border-indigo-300 shadow-sm'
                  : 'bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50'
              }`}
            >
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="font-mono text-slate-400 font-semibold text-[10px]">STAGE {stage.id}</span>
                <span className="font-mono text-indigo-600 text-[10px] flex items-center gap-1 font-semibold">
                  <Clock className="w-2.5 h-2.5" /> {stage.time}
                </span>
              </div>
              <div className={`text-xs font-bold line-clamp-1 ${isSelected ? 'text-indigo-900' : 'text-slate-800'}`}>
                {stage.title}
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Stage Deep-Dive Card */}
      <div className="quant-card p-6 rounded-2xl border border-slate-200 bg-white shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-4">
          <div>
            <div className="text-xs font-mono text-indigo-600 font-semibold">
              STAGE {selectedStage.id} OF 14
            </div>
            <h3 className="text-xl font-bold text-slate-900 mt-0.5">{selectedStage.title}</h3>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 border border-slate-200">
              Execution: {selectedStage.time}
            </span>
            <span className="px-2.5 py-1 rounded-md bg-emerald-50 text-emerald-700 border border-emerald-200 font-semibold">
              Status: Verified
            </span>
          </div>
        </div>

        {/* Mathematical Formulation */}
        <div className="space-y-2">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider font-mono">
            Mathematical Objective & Formal Specification
          </span>
          <div className="p-3.5 rounded-xl bg-slate-900 text-indigo-300 border border-slate-800 font-mono text-xs overflow-x-auto shadow-inner">
            <code>{selectedStage.formula}</code>
          </div>
        </div>

        {/* Description & Engineering Details */}
        <div className="space-y-2">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider font-mono">
            Algorithmic Details & Implementation Notes
          </span>
          <p className="text-xs text-slate-700 font-mono leading-relaxed bg-slate-50 p-4 rounded-xl border border-slate-200">
            {selectedStage.desc}
          </p>
        </div>

        {/* Navigation between stages */}
        <div className="flex justify-between items-center pt-2">
          <button
            onClick={() => setSelectedStageId((prev) => Math.max(1, prev - 1))}
            disabled={selectedStageId === 1}
            className="px-3.5 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-mono disabled:opacity-30 disabled:cursor-not-allowed border border-slate-200 transition-colors"
          >
            ← Previous Stage
          </button>
          <span className="text-xs font-mono text-slate-400">
            Stage {selectedStageId} of 14
          </span>
          <button
            onClick={() => setSelectedStageId((prev) => Math.min(14, prev + 1))}
            disabled={selectedStageId === 14}
            className="px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-mono disabled:opacity-30 disabled:cursor-not-allowed transition-colors shadow-sm font-semibold"
          >
            Next Stage →
          </button>
        </div>
      </div>
    </div>
  );
}
