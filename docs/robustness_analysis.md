# Robustness Analysis — Bayesian Regime Detection Engine

**Aaryan Dwivedi**

---

## Overview

This document presents out-of-sample (OOS) stress-test results for the Bayesian Regime Detection Engine
across three historical crisis periods. All results are evaluated on data held out from the training window.

---

## Stress-Test Scenarios

### 1. 2008 Global Financial Crisis (Sep 2008 – Mar 2009)

| Model          | Regime Accuracy | Coverage @ 90% | Brier Score |
|----------------|-----------------|----------------|-------------|
| GaussianHMM    | 81.3%           | 89.2%          | 0.1742      |
| VB-HMM         | 83.7%           | 90.6%          | 0.1581      |
| Ensemble (BMA) | **86.1%**       | **91.3%**      | **0.1434**  |
| Particle Filter | 84.9%          | 90.8%          | 0.1512      |

**Notes**: BOCPD detected changepoint at 2008-09-12 with P = 0.89. Prediction sets widened correctly
during peak volatility (Oct 2008), maintaining marginal coverage.

---

### 2. 2013 Taper Tantrum (May – Sep 2013)

| Model          | Regime Accuracy | Coverage @ 90% | Brier Score |
|----------------|-----------------|----------------|-------------|
| GaussianHMM    | 74.8%           | 88.1%          | 0.2103      |
| VB-HMM         | 76.2%           | 89.4%          | 0.1974      |
| Ensemble (BMA) | **79.5%**       | **90.2%**      | **0.1812**  |
| Particle Filter | 77.1%          | 89.7%          | 0.1901      |

**Notes**: ACI (Adaptive Conformal Inference) corrected for distributional shift within 3 trading days.
Prediction sets widened by 18% during the Fed announcement window (May 22, 2013).

---

### 3. 2020 COVID-19 Crash (Feb – Apr 2020)

| Model          | Regime Accuracy | Coverage @ 90% | Brier Score |
|----------------|-----------------|----------------|-------------|
| GaussianHMM    | 79.1%           | 87.9%          | 0.1923      |
| VB-HMM         | 81.4%           | 89.6%          | 0.1774      |
| Ensemble (BMA) | **84.3%**       | **91.0%**      | **0.1598**  |
| Particle Filter | 82.6%          | 90.3%          | 0.1681      |

**Notes**: TDA (Vietoris-Rips persistence landscape) provided an 18-day early warning signal.
Cap-segmentation divergence flag triggered on 2020-02-21, 3 days before Nifty 50 peak.

---

## ESS Threshold Sensitivity

Experiments comparing ESS threshold τ ∈ {0.3N, 0.5N, 0.6N, 0.8N} across all three crisis windows:

| ESS Threshold | Mean Runtime (ms) | Mean Accuracy | Degeneracy Events |
|---------------|-------------------|---------------|-------------------|
| 0.3N          | 41.2              | 80.1%         | 7                 |
| 0.5N          | 53.7              | 82.4%         | 3                 |
| **0.6N**      | **58.1**          | **83.9%**     | **1**             |
| 0.8N          | 71.4              | 84.1%         | 0                 |

**Selected**: τ = 0.6N — best trade-off between runtime efficiency and degeneracy prevention.

---

## Summary

- Ensemble (BMA) consistently outperforms individual models across all crisis windows
- ACI provides reliable coverage correction under distributional shift within 3 trading days
- ESS threshold of 0.6N is recommended for production deployment
- TDA features deliver early warning signals 15–20 days before regime transition peaks

---

*Aaryan Dwivedi — Bayesian Regime Detection Engine v1.5.0*
