# Changelog

All notable changes to the **Bayesian Regime Detection Engine** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

**Aaryan Dwivedi**

---

## [Unreleased]

### Planned
- Live NSE/BSE data feed integration via SEBI-compliant API
- Real-time regime probability dashboard (Streamlit)
- Automated model drift alerts via PSI monitoring pipeline
- GPU-accelerated particle filter using CuPy

---

## [1.5.0] — 2026-09-02

### Added
- Regime robustness tests: out-of-sample stress scenarios for 2008, 2013, 2020 crises
- `docs/robustness_analysis.md` — full OOS stress-test results with coverage tables

### Changed
- Improved particle filter resampling efficiency: ESS threshold tuned from 0.5N to 0.6N
- Updated CHANGELOG to reflect v1.5.0 milestone

### Fixed
- Edge case in `GaussianHMM.fit` when all observations fall in a single regime cluster

---

## [1.4.0] — 2026-08-30

### Added
- `docs/performance_benchmarks.md` — latency and throughput benchmarks across regime types
- Logging improvements: structured JSON log output for production pipelines

### Changed
- Tuned HMM transition priors for improved bull/bear/sideways classification accuracy

---

## [1.3.0] — 2026-08-26

### Added
- `.github/CONTRIBUTING.md` — project contribution guidelines, coding standards, and commit conventions

### Changed
- README: added "Last Updated August 2026" badge

---

## [1.2.0] — 2026-08-25

### Added
- `docs/last updated` badge in README reflecting August 2026

### Fixed
- Inline `build_regime_output` definition in `submission.py` to remove external dependency

---

## [1.1.0] — 2026-08-24

### Added
- Complete 14-stage pipeline update: ensembling, Brier decomposition, RPS skill score, IC artefact generation
- OOS prediction array alignment fix for HMM, Ensemble, and VB-HMM in Stage 10

### Fixed
- Log-domain forward–backward algorithm in `GaussianHMM` — eliminates log-likelihood underflow and NaN probabilities
- Removed deprecated `multi_class` parameter in `LogisticRegression`
- NaN observation sanitisation in `GaussianHMM.fit`
- Numerical Laplace smoothing and parameter sanitisation in `GaussianHMM` and `ParticleFilter`
- Non-ASCII character sanitisation for Windows CP1252 compatibility
- Robust quantile initialisation in `GaussianHMM`

---

## [1.0.0] — 2026-08-20

### Added
- `submission.py` — pure self-contained 12-stage Bayesian Regime Detection Engine
  - Zero external dependency fallbacks; all algorithms implemented from scratch
  - Baum-Welch HMM (custom implementation)
  - Variational Bayes HMM
  - Calibrated ensemble (BMA + constrained stacking)
  - Split / APS / Mondrian conformal prediction (exact 90% marginal coverage)
  - Walk-forward backtest: IR = +0.6142, DSR = 0.8741
- `src/data/synthetic_data.py` — 5-regime Student-t market simulation (2007–2024)
- `src/data/feature_engineering.py` — 30+ features: returns, vol, breadth, macro, flows, TDA, GCN, PSI
- `src/models/frequentist_hmm.py` — Gaussian HMM + BIC selection (K = 3/5/7) + duration statistics
- `src/models/bayesian_hmm.py` — PyMC + NUTS (4 chains × 2000 draws) + ArviZ diagnostics + WAIC/LOO
- `src/models/msm_baseline.py` — statsmodels `MarkovRegression` (K = 3) + BIC
- `src/models/foundation_models.py` — Chronos + TimesFM + `HybridRegimeModel` + sample efficiency curves
- `src/models/bayesian_dl.py` — MC Dropout + VI BNN + Deep Ensemble + SHAP + ECE + reliability diagram
- `src/models/rs_var.py` — Bayesian RS-VAR + IRF + NumPyro alternative + WAIC/LOO
- `src/inference/particle_filter.py` — Bootstrap SIR, 5000 particles, systematic resampling, ESS diagnostics
- `src/inference/bocpd.py` — BOCPD (Normal-Gamma, Student-t predictive) + crisis validation + streaming Dirichlet
- `src/calibration/conformal.py` — Split/APS/Mondrian/ACI/CQR + ECE + Brier + RPS + rolling coverage
- `src/ensembling/ensembling.py` — BMA + constrained stacking + WAIC/LOO weighting + output contract
- `src/backtest/backtest.py` — Information Ratio + tracking error + IC artefact + Kelly overlay
- `R/models.R` — depmixS4, MSwM, rstanarm, brms, bcp, PELT + conformal + reconciliation
- `R/conformal.R` — Split/APS/Mondrian/ACI + ECE/RPS/Brier + reliability diagram
- `R/reconciliation.R` — Python vs R cross-validation (correlation/MAE/RMSE per regime)
- `R/stan_hmm.stan` — Stan HMM with forward algorithm + Viterbi decoding
- `docs/report.md` — 40+ page technical report (14 sections, 5 Indian case studies)
- `docs/presentation.md` — 18-slide deck
- `docs/model_card.md` — model specification, output contract, MCMC diagnostics, calibration table
- `CHECKLIST.md` — complete requirements verification
- `requirements.txt` — full dependency specification

### Key Metrics at v1.0.0
| Metric | Value |
|---|---|
| Information Ratio (walk-forward) | **0.61** |
| Tracking Error (ann.) | 2.1% |
| Calibration ECE | **0.0156** |
| 95% VaR (1-year MC) | -12.3% |
| Deflated Sharpe Ratio | **0.8741** |
| Coverage @ 90% target | **90.7%** / **91.2%** |

---

*Aaryan Dwivedi*

## [1.4.1] - 2026-08-31
### Fixed
- Minor typo corrections in documentation

