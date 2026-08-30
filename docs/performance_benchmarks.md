# Performance Benchmarks - Bayesian Regime Detection Engine

**Author:** Aaryan Dwivedi
**Last Updated:** 2026-08-30

---

## Overview

This document reports latency and throughput benchmarks for the Bayesian Regime Detection Engine across different regime types and dataset sizes. All benchmarks were run on a standard CPU environment (Intel Core i7, 16 GB RAM) unless noted otherwise.

---

## Benchmark Environment

| Property | Value |
|---|---|
| CPU | Intel Core i7-11800H |
| RAM | 16 GB DDR4 |
| OS | Windows 11 |
| Python | 3.11.x |
| NumPy | 1.26.x |
| SciPy | 1.13.x |

---

## Latency Benchmarks

### HMM Inference (Viterbi Decoding)

| Dataset Size (days) | Latency (ms) | Throughput (samples/s) |
|---|---|---|
| 252 (1 year) | 4.2 | 59,952 |
| 1,260 (5 years) | 18.7 | 67,379 |
| 2,520 (10 years) | 36.1 | 69,806 |
| 5,040 (20 years) | 72.8 | 69,231 |

### Bayesian Particle Filter

| Particles | Dataset Size (days) | Latency (ms) |
|---|---|---|
| 500 | 252 | 12.3 |
| 1,000 | 252 | 23.8 |
| 2,000 | 252 | 47.1 |
| 1,000 | 1,260 | 118.4 |

---

## Regime Classification Accuracy

| Regime Type | Precision | Recall | F1-Score |
|---|---|---|---|
| Bull | 0.89 | 0.91 | 0.90 |
| Bear | 0.87 | 0.84 | 0.85 |
| Sideways | 0.76 | 0.79 | 0.77 |
| Crisis | 0.93 | 0.88 | 0.90 |
| **Macro Avg** | **0.86** | **0.86** | **0.86** |

---

## Memory Usage

| Component | Peak RAM (MB) |
|---|---|
| HMM training (10y data) | 48 |
| Particle filter (1000p) | 31 |
| Feature engineering pipeline | 22 |
| Full pipeline (end-to-end) | 112 |

---

## Notes

- Benchmarks represent single-threaded CPU execution. Parallel execution via joblib can yield ~3-4x speedup on multi-core systems.
- GPU-accelerated particle filtering (CuPy) is planned for v1.5.0 and is expected to yield 10-20x latency reduction.
- All timings are median over 100 runs to reduce variance from OS scheduling noise.
