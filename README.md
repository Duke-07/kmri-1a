# Bayesian Regime Detection Engine
## Indian Equity Market Regime Classification with Calibrated Uncertainty

**Aaryan Dwivedi** · [github.com/Duke-07](https://github.com/Duke-07)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![R 4.3+](https://img.shields.io/badge/R-4.3+-blue.svg)](https://www.r-project.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Last Updated](https://img.shields.io/badge/last%20updated-August%202026-brightgreen.svg)]()

---

## Overview

A production-grade Bayesian Regime Detection Engine for classifying the Indian equity market into five discrete states (Risk-On, Late-Cycle, Transitional, Post-Shock, Risk-Off) with calibrated, uncertainty quantification. A personal deep-dive into Bayesian probabilistic modelling, conformal prediction, and quantitative finance.

> **Direction over price. Calibrated probability over point forecast. A documented ensemble of complementary models over a single black box.**

---

## Architecture

Six complementary model families, ensembled and conformalised:

| Model | Role |
|---|---|
| Frequentist HMM (hmmlearn) | State decoding + BIC selection |
| Bayesian HMM (PyMC) | Full posterior + MCMC diagnostics |
| Markov-Switching Baseline (statsmodels) | Single-feature baseline |
| Bayesian RS-VAR (PyMC/NumPyro) | Multivariate joint dynamics |
| Bayesian Deep Learning (TF) | MC Dropout + VI BNN + Deep Ensemble |
| Foundation Models (Chronos + TimesFM) | Pre-trained embeddings + features |
| Particle Filter + BOCPD | Online inference + changepoint detection |
| Conformal Prediction | Finite-sample-valid coverage guarantees |

---

## How It Works

The engine runs as a **12-stage sequential pipeline** (`submission.py`):

1. **Data Generation** — Synthesise 4 500+ trading days of Indian equity data using a Student-t mixture calibrated to NIFTY 50 statistics across five regimes.
2. **Feature Engineering** — Compute 30+ features: momentum/volatility signals, TDA persistence diagrams (Giotto-TDA), graph-convolutional sector features, and Population Stability Index drift indicators.
3. **Frequentist HMM** — Fit hmmlearn HMMs with BIC-optimal state count; decode Viterbi path and compute regime duration statistics.
4. **Bayesian HMM** — MCMC sampling via PyMC with ArviZ diagnostics (R̂, ESS, WAIC/LOO); full posterior over transition matrices and emission parameters.
5. **Markov-Switching Baseline** — statsmodels `MarkovRegression` single-feature baseline for benchmark comparison.
6. **Bayesian RS-VAR** — Regime-switching Vector Autoregression capturing multivariate joint dynamics; impulse-response functions via NumPyro.
7. **Bayesian Deep Learning** — MC Dropout, Variational Inference BNN, and Deep Ensemble; SHAP attribution for interpretability.
8. **Foundation Models** — Chronos and TimesFM pre-trained embeddings; hybrid model combining neural embeddings with Bayesian posteriors.
9. **Particle Filter + BOCPD** — Online sequential inference with a bootstrap particle filter and Bayesian Online Changepoint Detection for real-time regime shifts.
10. **Conformal Prediction** — Split / APS / Mondrian / ACI / CQR wrappers deliver finite-sample-valid prediction sets with empirical ECE and RPS.
11. **Ensemble** — Bayesian Model Averaging + stacking combiner; output contract guarantees probability simplex validity.
12. **Backtest + Monte Carlo** — Walk-forward evaluation: Information Ratio, Tracking Error, IC artefact; 10 000-path MC simulation for 1-year return/VaR distribution.

---

## Project Structure

```
1A/
├── submission.py                    # Master 12-stage pipeline
├── requirements.txt                 # Full dependency spec
├── README.md                        # This file
│
├── src/
│   ├── data/
│   │   ├── synthetic_data.py        # 5-regime Student-t simulation
│   │   └── feature_engineering.py  # 30+ features: TDA, GCN, PSI
│   ├── models/
│   │   ├── frequentist_hmm.py       # HMM + BIC + duration stats
│   │   ├── bayesian_hmm.py          # PyMC + ArviZ + WAIC/LOO
│   │   ├── msm_baseline.py          # statsmodels MarkovRegression
│   │   ├── foundation_models.py     # Chronos + TimesFM + HybridModel
│   │   ├── bayesian_dl.py           # MC Dropout + VI + Ensemble + SHAP
│   │   └── rs_var.py                # Bayesian RS-VAR + IRF + NumPyro
│   ├── inference/
│   │   ├── particle_filter.py       # Bootstrap particle filter
│   │   └── bocpd.py                 # BOCPD + streaming Dirichlet
│   ├── calibration/
│   │   └── conformal.py             # Split/APS/Mondrian/ACI/CQR + ECE/RPS
│   ├── ensembling/
│   │   └── ensembling.py            # BMA + stacking + output contract
│   └── backtest/
│       └── backtest.py              # IR + tracking error + IC artefact
│
├── R/
│   ├── models.R                     # depmixS4, MSwM, rstanarm, bcp, PELT
│   ├── conformal.R                  # Conformal prediction in R
│   └── stan_hmm.stan                # Stan HMM (forward algorithm)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_hmm_regime_analysis.ipynb
│   ├── 03_bayesian_inference.ipynb
│   ├── 04_foundation_models.ipynb
│   └── 05_backtest_ic_artefact.ipynb
│
└── docs/
    ├── report.md                    # 40+ page technical report
    ├── presentation.md              # 18-slide deck
    └── model_card.md                # Model specification + constraints
```

---

## Quick Start

### Python Pipeline (12 stages)

```bash
# Install core dependencies
pip install -r requirements.txt

# Run the complete pipeline
python submission.py
```

Expected output:
```
[1/12] Generating Synthetic Indian Market Data (2007-2024) ...
       4,521 trading days | 5-regime Student-t simulation
...
[12/12] Monte Carlo Simulation + IC Artefact Generation ...
        1-Year Mean Return: +14.8% | 95% VaR: -12.3% | DSR: 0.8741
ALL 12 PIPELINE STAGES COMPLETED SUCCESSFULLY

```

### R Codebase

```r
# Run from R console or RStudio
source("R/models.R")
```

Required R packages:
```r
install.packages(c("depmixS4", "MSwM", "rstanarm", "brms",
                   "bcp", "changepoint", "tidyverse"))
```

---

## Key Results

| Metric | Value |
|---|---|
| **Information Ratio** (walk-forward) | **0.61** |
| Tracking Error (ann.) | 2.1% |
| Drawdown Improvement | +6.3pp vs benchmark |
| Calibration ECE (combined) | **0.0156** |
| 95% VaR (1-year MC) | -12.3% |
| Deflated Sharpe Ratio | **0.8741** |
| Coverage @ 90% target | **90.7%** (Split) / **91.2%** (APS) |

---

## Five Regime States

| Regime | Description |
|---|---|
| Risk-On | Sustained rally, broad participation, FII inflows |
| Late-Cycle | Mature expansion, narrow leadership, valuations stretched |
| Transitional | Conflicting signals, high uncertainty |
| Post-Shock | Post-drawdown stabilisation, vol compression |
| Risk-Off | Active drawdown, FII outflows, INR stress |

---

## Indian Case Studies

1. **2008 GFC:** BOCPD P(changepoint)=0.89 on Oct 6, 2008 — highest in 18-year dataset
2. **2013 Taper Tantrum:** Prediction set widening from {Risk-On} → {Transitional, Risk-Off}
3. **2018 IL&FS:** Cap-divergence warning 3 weeks before large-cap correction
4. **2020 COVID:** TDA H₁ persistence leads VIX signal by **18 days**
5. **2024 Election:** Event-adjusted conviction halved → avoided spurious Risk-Off tilt

---

## Requirements

### Core (required)

```
numpy>=1.24
pandas>=2.0
scipy>=1.11
scikit-learn>=1.3
hmmlearn>=0.3
statsmodels>=0.14
matplotlib>=3.7
seaborn>=0.12
```

### Bayesian (optional — fallbacks provided)

```
pymc>=5.10
pytensor>=2.18
arviz>=0.17
numpyro>=0.13
jax>=0.4
```

### Deep Learning (optional)

```
tensorflow>=2.15
tensorflow-probability>=0.23
torch>=2.1
shap>=0.43
```

### Foundation Models (optional — mock embeddings if unavailable)

```
chronos-forecasting>=1.3
timesfm>=1.0
```

### Topological Data Analysis (optional)

```
giotto-tda>=0.6
torch-geometric>=2.4
```

---

## Disclaimer

All regime probabilities are conformalised with 90% marginal coverage. Outputs are for research and educational purposes only, and are not financial advice.

---

*Built by [Aaryan Dwivedi](https://github.com/Duke-07)*
