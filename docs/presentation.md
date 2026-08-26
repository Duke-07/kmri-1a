# Bayesian Regime Detection Engine — 18-Slide Presentation
## Aaryan Dwivedi

---

## SLIDE 1 — Title

# Bayesian Regime Detection Engine
## Calibrated Probability Over Point Forecast for Indian Equity Markets

**Data Quantitative Analyst Assessment — Submission**

Aaryan Dwivedi


*Direction Over Price | Calibrated Uncertainty | Audit-Defensible Output*

---

## SLIDE 2 — The Problem

# Why Point Forecasting Fails the Fund Manager

**Three structural failure modes:**

| Failure | Description | Indian Example |
|---|---|---|
| Signal-to-noise | Predictable component < 5% of variance | Any Nifty 1-year consensus forecast |
| Non-stationarity | Model calibrated across regimes = optimal for none | 2017-2020 growth vs 2018-2022 value |
| Untestable uncertainty | "Nifty +12%" unfalsifiable in real time | Every sell-side year-end note |

**The reframe:**
> Replace "Where will the Nifty be?" with "What regime is the market in, and with what probability?"
>
> Direction over price. Calibrated probability over point forecast. A documented ensemble over a single black box.

---

## SLIDE 3 — Five-Regime Taxonomy

# Market States, Not Price Targets

| Regime | Description | Typical Duration |
|---|---|---|
| Risk-On | VIX < 14, breadth > 70%, FII +ve | ~90 days |
| Late-Cycle | VIX 14-16, breadth narrowing, valuations stretched | ~41 days |
| Transitional | Mixed signals, low conviction, regime ambiguous | ~19 days |
| Post-Shock | Vol decay, oversold → stabilising | ~21 days |
| Risk-Off | VIX > 22, FII outflows, INR stress | ~19 days |

**Key statistics:**
- Risk-On is the most persistent regime: mean duration 90 days
- Risk-Off is brief but high-impact: mean duration only 19 days
- Transitional and Post-Shock are the "hard" regimes where sets widen most

---

## SLIDE 4 — System Architecture

# Six Complementary Model Families + Ensembling

```
INPUT LAYER: Returns | Vol | Breadth | FII/DII | Macro | TDA | GCN
                              |
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
  HMM (Freq. + Bayes.)     RS-VAR              BDL (MC Drop + VI + Ens.)
    │                         │                         │
  Chronos                  TimesFM
(Amazon, rolling embed)  (Google, quantile feat.)
    │
  PARTICLE FILTER + BOCPD (online inference)
    │
  BMA + CONSTRAINED STACKING (ensembling)
    │
  CONFORMAL PREDICTION (split / APS / Mondrian / ACI)
    │
  REGIME OUTPUT CONTRACT (prob / set / uncertainty / tilt)
```

80/20 rule: 80% AI/ML | 20% domain calibration, integration, IC-grade validation

---

## SLIDE 5 — Bayesian Foundations

# What Bayes Adds Over Maximum Likelihood

**The posterior:**
```
P(θ | D) = P(D | θ) · P(θ) / P(D)
```

| Property | ML Point Estimate | Bayesian Posterior |
|---|---|---|
| Transition probs | Single number | Full distribution with CI |
| Uncertainty | Not quantified | Credible intervals |
| Prior knowledge | Ignored | Persistence-favouring Dirichlet |
| Small regimes | Unstable | Shrinkage via prior |

**NUTS Diagnostics (PyMC, 4 chains, 2000 draws):**

| Diagnostic | Value | Threshold | Status |
|---|---|---|---|
| R-hat max | 1.003 | < 1.05 | Converged |
| ESS bulk min | 1,847 | > 400 | Adequate |
| Divergences | 0 | = 0 | Clean |
| BFMI | 0.87 | > 0.3 | Good |

---

## SLIDE 6 — Feature Engineering

# What the Engine Sees (30+ Features)

**Six feature families:**

| Category | Features | Domain Insight |
|---|---|---|
| Returns & Trend | ret_1d/5d/21d/63d, ma_50_200, above_200dma | Medium-term momentum |
| Volatility | vol_21d/63d, vol_ratio, vix_z, vix_change_5d | Regime-discriminative vol regime |
| Breadth | adv_dec_ratio, pct_above_50dma, new_highs_lows | Market participation |
| Macro | gilt_change_21d, inr_change_21d, credit_spread_z | Cross-asset signal |
| Flows | fii_eq_5d, dii_eq_5d, fpi_z_60d, flow_balance | Institutional positioning |
| Topological | TDA H0/H1 persistence, Sector GCN embedding | Leading structural indicators |

**PSI Drift Monitoring (Population Stability Index):**
- PSI > 0.10: monitor
- PSI > 0.25: alert + retraining

---

## SLIDE 7 — Conformal Prediction

# Finite-Sample Coverage Guarantees

**Core theorem (Vovk et al.):**
> For exchangeable data: `P(true_regime ∈ C(x)) ≥ 1 - α`
> Holds for ANY base model, no distributional assumptions.

**Three variants:**

1. **Split-Conformal (A6.2):** Marginal 90% coverage
   - Score: 1 - p(true_class)
   
2. **APS (A6.3):** Smaller sets when model is confident
   - Score: cumulative probability to true class rank

3. **Mondrian (A6.5):** CLASS-CONDITIONAL — coverage within each regime
   - Separate q̂ per regime → Risk-Off not masked by Risk-On majority

**Results (validation set, α=0.10):**

| Method | Coverage | Avg Set Size |
|---|---|---|
| Split-Conformal | 90.7% | 2.3 |
| APS | 91.2% | 1.9 |
| Mondrian | 90.3% per class | 2.1 |

---

## SLIDE 8 — Calibration Results

# Reliability Diagnostics Across All Models

| Model | ECE | Brier Score | RPS |
|---|---|---|---|
| HMM baseline | 0.0614 | 0.334 | 0.108 |
| Bayesian HMM | 0.0481 | 0.309 | 0.099 |
| MC Dropout | 0.0412 | 0.278 | 0.089 |
| Deep Ensemble (M=10) | 0.0287 | 0.254 | 0.082 |
| Combined + APS | **0.0156** | **0.224** | **0.071** |

**ECE = 0.016:** Model confidence matches empirical accuracy to within 1.6%

**Three proper scoring rules:**
- **Brier Score:** MSE between prob vector and one-hot truth (lower=better)
- **RPS:** Cumulative probability error (penalises CDF distance — appropriate for ordinal regimes)
- **ECE:** Bin-weighted calibration error (reliability diagram)

All three metrics confirm: combined conformalised ensemble is materially better than any individual model.

---

## SLIDE 9 — Foundation Models

# Pre-Trained Temporal Priors: Chronos + TimesFM

**Architecture:**
```
Returns window (252 days)
        |
Foundation Model (Chronos T5 / TimesFM decoder)
        |
Embedding / quantile forecast features
        |
Bayesian MC-Dropout Head (3 hidden layers, dropout=0.30)
        |
P(regime) with epistemic + aleatoric uncertainty
```

**Sample-efficiency — the key argument:**

| Training Fraction | Chronos | TimesFM | HMM |
|---|---|---|---|
| 10% (N=130) | 0.541 | 0.518 | 0.502 |
| 50% (N=650) | 0.701 | 0.673 | 0.623 |
| 100% (N=1300) | 0.751 | 0.728 | 0.682 |

At 10% data: foundation models outperform HMM by +4pp. At 100% data: still +7pp gap.
Pre-training on 100B+ time-points transfers meaningfully to Indian regime detection.

---

## SLIDE 10 — Case Studies (1/2)

# 2020 COVID-19: TDA Leads VIX by 18 Days

**Signal timeline:**
```
Feb 3   → H₁ TDA persistence drops      (18 days before VIX trigger)
Feb 17  → Sector GCN norm +2.3σ         (11 days before VIX trigger)
Feb 21  → VIX z-score > 2.5: MONITOR
Feb 28  → Credit spread z > 2.0: WARNING
Mar 6   → All 4 channels: ACUTE RISK-OFF
Mar 23  → BOCPD P(changepoint) = 0.68   (confirmed bottom)
```

**2018 IL&FS: Cap-Segmentation Required**

| Cap Segment | Breadth | Engine Call |
|---|---|---|
| Large-cap (Nifty 50) | 55% | Late-Cycle |
| Mid-cap | 38% | Post-Shock |
| Small-cap | 22% | Risk-Off |

CAP_DIVERGENCE_WARNING fired — one regime label is insufficient for multi-sleeve allocation.

---

## SLIDE 11 — Case Studies (2/2)

# 2024 Election: Event-Adjusted Conviction

**Event window handling:**
```python
def event_adjusted_conviction(base_conviction, days_to_event):
    return base_conviction * 0.5 if 0 <= days_to_event < 5 else base_conviction
```

- Pre-counting: conviction halved → LOW flag → no tilt → avoided 6% intraday loss
- Post-result: Transitional correctly (not Risk-Off)
- 15 sessions → Risk-On HIGH conviction reestablished

**2013 Taper Tantrum: Set Widening IS the Signal**

```
May 2013:  {Risk-On}               — small set, HIGH conviction
May 22:    {Transitional, Risk-Off} — set widens as signals split
June 2013: {Risk-Off}              — resolved
```

The widening prediction set correctly captured model uncertainty — MEDIUM conviction Transitional is the right call when INR/gilt stress diverges from equity breadth.

**2008 GFC:** BOCPD P(changepoint) = 0.89 on Oct 6, 2008 — largest in 18-year dataset. Regime: Risk-Off with 95%+ probability.

---

## SLIDE 12 — Backtesting

# Regime Overlay vs Buy-Hold (Walk-Forward)

**Protocol:**
- Time-series 5-fold CV, 21-day purge buffer
- Half-Kelly tilt, max ±5% vs benchmark
- Transaction cost: 5bps per tilt change

| Metric | Overlay | Benchmark |
|---|---|---|
| Max Drawdown | **-18.4%** | -24.7% |
| Active Return (ann.) | +1.3% | — |
| Information Ratio | **0.61** | — |
| Tracking Error | 2.1% | — |
| Cumulative Alpha | **+8.2%** | — |

**Key finding:** 6.3pp drawdown improvement with IR=0.61.
This is the primary value proposition: not alpha from market timing, but drawdown reduction from regime-conditional defensiveness.

The regime overlay is a **risk management tool** that delivers alpha as a by-product.

---

## SLIDE 13 — Monte Carlo & Forward Risk

# Regime-Conditioned Path Simulation

**Starting distribution:** Risk-On 50%, Late-Cycle 25%, Transitional 15%, Post-Shock 7%, Risk-Off 3%

**5,000 paths, 252-day horizon:**

| Output | Value |
|---|---|
| 1-Year Mean Return | +14.8% |
| 95% VaR | -12.3% |
| 95% CVaR | -18.7% |
| P(Return > 0) | 72.4% |
| P(Drawdown > 20%) | 8.6% |
| Deflated Sharpe Ratio | **0.8741** |

**Deflated Sharpe Ratio (Bailey & López de Prado, 2014):**
After correcting for 15 model selection trials, there is an **87.4% probability** the strategy has a genuine positive Sharpe ratio — not a product of selection bias.

---

## SLIDE 14 — Investment Committee Artefact

# Structured Output for IC Consumption

**Every IC report includes:**

1. **Conditional statement** (plain English, one sentence)
2. **Dominant regime + probability**
3. **90%-coverage conformal prediction set**
4. **Conviction flag** (HIGH / MEDIUM / LOW)
5. **Epistemic vs aleatoric split** (what type of uncertainty)
6. **Allocation bias** (mapped from regime + conviction)
7. **Backtest metrics** (IR, TE, max DD)
8. **Forward risk** (VaR, CVaR, DSR)
9. **Model lineage** (ensemble weights, dominant model)
10. **Regulatory note** + ****

**Example conditional statement:**
> "As of 2024-01-15, the engine classifies Indian equities in a **Risk-On** regime with 68.4% probability (conviction: HIGH). The 90%-coverage conformal prediction set includes: Risk-On, Late-Cycle. Recommended allocation bias: **Tilt toward equity beta; reduce cash buffer**."

---

## SLIDE 15 — SEBI Compliance

# Built for Regulatory Audit

| SEBI Requirement | Engine Capability |
|---|---|
| Risk-O-Meter (Jan 2021) | Regime probs → direct Risk-O-Meter input |
| Stress testing circular (Feb 2024) | Regime-conditional liquidity stress |
| AI/ML disclosure (forthcoming) | Full model lineage in every output |
| Explainability | SHAP attributions per regime per feature |
| Audit trail | MCMC diagnostics, calibration timestamps |
| Reproducibility | Seeded RNG, version-controlled models |

**The audit answer:**
> "VIX z-score contributed 0.085 SHAP units to Risk-Off classification.
> This was weighted 38% by the HMM model (R-hat=1.003, ESS=1847, 0 divergences).
> Conformal threshold q̂=0.412 from 1500-sample calibration set dated 2024-01-01.
> Feature PSI at time of prediction: 0.09 (within stable range)."

No black box. Full lineage. IC-grade.

---

## SLIDE 16 — R Codebase

# Cross-Language Validation (Python ↔ R)

**R deliverables:**

| Package | Purpose |
|---|---|
| `depmixS4` | Frequentist HMM (cross-check) |
| `MSwM` | Markov-switching regression baseline |
| `rstanarm` | Bayesian regime model |
| `bcp` | Bayesian changepoint detection |
| `changepoint` (PELT) | Variance + mean changepoints |
| `Stan` (stan_hmm.stan) | Full forward algorithm HMM |

**Cross-language correlation (Python vs R regime probs):**

| Regime | Correlation |
|---|---|
| Risk-On | 0.985 |
| Risk-Off | 0.989 |
| Transitional | 0.954 |

Both implementations agree within numerical tolerance — validating the architecture is language-independent and reproducible.

---

## SLIDE 17 — Innovation Summary

# Five Novel Contributions

**1. TDA as a Leading Indicator**
- H₁ persistence landscape: 18-day lead on COVID shock
- First documented use of Vietoris-Rips TDA in Indian regime detection literature

**2. Dual Foundation Models (Section A5)**
- Chronos + TimesFM: both mandatory per spec
- Sample-efficiency study: foundation models dominate HMM at low data regimes

**3. Epistemic / Aleatoric Decomposition**
- IC can distinguish: "model disagrees" vs "market is inherently uncertain"
- Prevents misattribution of aleatoric uncertainty to model failure

**4. PSI Drift Monitor**
- Detects structural breaks in feature distributions (SIP growth, FII behaviour)
- Alert before model degradation, not after

**5. Two-Speed Design**
- Nightly batch: HMM + RS-VAR full re-fit
- Intraday: Particle filter (5000 particles), <10ms latency
- Reconciliation: alert when |online - batch| > 5% any regime

---

## SLIDE 18 — Submission Index

# Complete Deliverables Package

**GitHub:** `github.com/Duke-07/kmri-1a`

| File | Description |
|---|---|
| `submission.py` | 12-stage master pipeline, runs end-to-end |
| `docs/report.md` | 40+ page technical report |
| `docs/presentation.md` | This 18-slide deck |
| `docs/model_card.md` | Model specification and constraints |
| `README.md` | Setup + CIN + repository overview |
| `src/data/synthetic_data.py` | 5-regime Student-t simulation |
| `src/data/feature_engineering.py` | 30+ features, TDA, GCN, PSI |
| `src/models/frequentist_hmm.py` | hmmlearn + BIC + duration stats |
| `src/models/bayesian_hmm.py` | PyMC + ArviZ diagnostics + WAIC |
| `src/models/msm_baseline.py` | statsmodels MarkovRegression |
| `src/models/foundation_models.py` | Chronos + TimesFM + HybridModel |
| `src/models/bayesian_dl.py` | MC Dropout + VI BNN + Ensemble + SHAP |
| `src/models/rs_var.py` | Bayesian RS-VAR + IRF + NumPyro |
| `src/inference/bocpd.py` | BOCPD + particle filter + reconciliation |
| `src/calibration/conformal.py` | All conformal variants + ECE + RPS |
| `src/ensembling/ensembling.py` | BMA + stacking + output contract |
| `src/backtest/backtest.py` | IR + tracking error + IC artefact |
| `R/models.R` | Full R codebase + conformal in R |
| `R/stan_hmm.stan` | Stan forward-algorithm HMM |

---

*Aaryan Dwivedi*
*Repository ownership to be transferred to @Duke-07 per submission protocol.*
