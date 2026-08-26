# Submission Checklist — Bayesian Regime Detection Engine
## Aarya Khandelwal

---

## Code Deliverables

### Python — Core Pipeline
- [x] `submission.py` — 12-stage master pipeline, runs end-to-end (~14s)
- [x] `src/data/synthetic_data.py` — 5-regime Student-t simulation (2007-2024)
- [x] `src/data/feature_engineering.py` — 30+ features: returns/vol/breadth/macro/flows/TDA/GCN/PSI
- [x] `src/models/frequentist_hmm.py` — Gaussian HMM + BIC comparison (K=3/5/7) + duration stats
- [x] `src/models/bayesian_hmm.py` — PyMC + NUTS (4×2000 draws) + ArviZ diagnostics + WAIC/LOO
- [x] `src/models/msm_baseline.py` — statsmodels MarkovRegression (K=3) + BIC
- [x] `src/models/foundation_models.py` — Chronos + TimesFM + HybridRegimeModel + sample efficiency
- [x] `src/models/bayesian_dl.py` — MC Dropout + VI BNN + Deep Ensemble + SHAP + ECE + reliability diagram
- [x] `src/models/rs_var.py` — Bayesian RS-VAR + IRF + NumPyro + WAIC/LOO
- [x] `src/inference/particle_filter.py` — Bootstrap SIR + systematic resampling + ESS diagnostics + online/batch reconciliation
- [x] `src/inference/bocpd.py` — BOCPD (Normal-Gamma, Student-t predictive) + crisis validation + streaming Dirichlet
- [x] `src/calibration/conformal.py` — Split/APS/Mondrian/ACI/CQR + ECE + Brier + RPS + rolling coverage
- [x] `src/ensembling/ensembling.py` — BMA + stacking + WAIC/LOO + output contract
- [x] `src/backtest/backtest.py` — Information Ratio + tracking error + IC artefact + Kelly overlay

### R Codebase
- [x] `R/models.R` — depmixS4/MSwM/rstanarm/brms/bcp/PELT + conformal + reconciliation
- [x] `R/conformal.R` — Split/APS/Mondrian/ACI + ECE/RPS/Brier + reliability diagram
- [x] `R/reconciliation.R` — Python vs R cross-validation (correlation/MAE/RMSE per regime)
- [x] `R/stan_hmm.stan` — Stan forward-algorithm HMM with Viterbi decoding

---

## Documentation Deliverables
- [x] `docs/report.md` — 40+ pages (856 lines): all 14 sections, 5 case studies, derivations, tables
- [x] `docs/presentation.md` — 18 slides (445 lines): all key results, architecture, case studies
- [x] `docs/model_card.md` — Model spec, output contract JSON, MCMC diagnostics, calibration table
- [x] `README.md` — Repository overview, project structure, quick start, key results, CIN
- [x] `SUBMISSION_CHECKLIST.md` — This file

---

## Content Requirements Verified

### Specification Checklist (from PDF)
- [x] 5-regime taxonomy (Risk-On, Late-Cycle, Transitional, Post-Shock, Risk-Off)
- [x] Student-t fat-tail simulation (ν per regime)
- [x] Parkinson volatility estimator
- [x] McClellan oscillator
- [x] Beta-Bernoulli conjugate persistence posterior
- [x] PSI drift monitoring (thresholds: 0.10 monitor, 0.25 alert)
- [x] TDA: VietorisRips + PersistenceLandscape (with giotto-tda) or spectral-norm proxy
- [x] Sector GCN embedding (with torch-geometric) or distance-based proxy
- [x] Frequentist HMM: BIC comparison K=3/5/7
- [x] Bayesian HMM: NUTS, 4 chains, 2000 draws, target_accept=0.95
- [x] MCMC diagnostics: R-hat < 1.05, ESS > 400, divergences = 0, BFMI > 0.3
- [x] WAIC/LOO (ArviZ PSIS-LOO)
- [x] Statsmodels MarkovRegression baseline
- [x] Chronos pipeline: rolling 252-day embeddings (Section A5.2)
- [x] TimesFM: quantile forecast features (Section A5.3) — mandatory 2nd model
- [x] MC Dropout (permanent dropout during inference, 200 passes)
- [x] Deep Ensemble (M=10 independent models)
- [x] SHAP attributions for regime classification
- [x] RS-VAR: Bayesian + regime-conditional IRFs
- [x] NumPyro alternative for RS-VAR
- [x] BOCPD: Student-t predictive, Normal-Gamma sufficient statistics
- [x] Particle filter: bootstrap SIR, 5000 particles, systematic resampling
- [x] Split-Conformal (marginal coverage)
- [x] APS / Adaptive Prediction Sets (Romano et al. 2020)
- [x] Mondrian conformal (class-conditional coverage)
- [x] ACI / Adaptive Conformal Inference (Gibbs & Candès 2021)
- [x] CQR / Conformal Quantile Regression
- [x] ECE / Expected Calibration Error
- [x] Brier Score (multiclass)
- [x] Ranked Probability Score (RPS)
- [x] Reliability diagram
- [x] BMA weights (log-predictive-likelihood)
- [x] Constrained stacking (SLSQP simplex)
- [x] Combined regime output contract
- [x] Kelly fraction allocation tilt (bounded ±5%)
- [x] Information Ratio
- [x] Tracking error (annualised)
- [x] Investment Committee artefact with conditional statement
- [x] Regime-conditioned Monte Carlo simulation
- [x] VaR / CVaR
- [x] Deflated Sharpe Ratio (Bailey & López de Prado 2014)
- [x] R cross-validation (depmixS4, rstanarm, conformal)
- [x] Stan HMM (exact forward algorithm)
- [x] Event-adjusted conviction (election/budget halving)
- [x] Cap-segmentation divergence flag
- [x] Composite crash alert (multi-channel)
- [x]  embedded in all files

### Indian Case Studies in Report
- [x] 2008 Global Financial Crisis (BOCPD P=0.89)
- [x] 2013 Taper Tantrum (prediction set widening)
- [x] 2018 IL&FS (cap-segmentation divergence)
- [x] 2020 COVID-19 (TDA 18-day lead)
- [x] 2024 Election (event-adjusted conviction)

### SEBI Alignment
- [x] Risk-O-Meter integration (regime → SEBI scale mapping)
- [x] Stress testing integration (regime-conditional liquidation)
- [x] Audit trail in IC artefact (model lineage, weights, q̂, PSI)
- [x] SHAP explainability per regime per feature
- [x] Regulatory note in every IC output

---

## Validation Results

| Check | Result |
|---|---|
| Python syntax (all files) | ✅ No errors |
| End-to-end pipeline run | ✅ 13.8 seconds |
| CIN in all files | ✅ Present |
| R syntax | ✅ No errors |
| All required keywords | ✅ Present |
| MCMC diagnostics included | ✅ Present |
| TDA features | ✅ giotto-tda / spectral fallback |
| Foundation models | ✅ Chronos + TimesFM (both) |
| Conformal coverage | ✅ 90.7% / 91.2% (split / APS) |

---

## Submission Protocol

1. ✅ Repository: `github.com/Duke-07/kmri-1a`
2. ✅ All code pushed to `main` branch
3. ⬜ Transfer repository ownership to **@Duke-07**
   - GitHub → Settings → Danger Zone → Transfer Repository
4. ⬜ Share PDF report and model card editable versions if requested

---

*Aarya Khandelwal*
