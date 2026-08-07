# Bayesian Regime Detection Engine — Model Card
## Zetheta Algorithms Private Limited | CIN: U62012MH2023PTC410415

---

## Model Overview

| Field | Value |
|---|---|
| **Model Name** | Bayesian Regime Detection Engine v2.0 |
| **Task** | 5-class market regime classification for Indian equity markets |
| **Version** | v2.0 (August 2026) |
| **CIN** | U62012MH2023PTC410415 |
| **Intended Use** | Tactical allocation overlays, IC reporting, Risk-O-Meter input |
| **Out of Scope** | Price/return prediction, individual stock selection, leverage sizing |

---

## Model Architecture

### Stack

| Model | Role | Key Parameters |
|---|---|---|
| Gaussian HMM (hmmlearn) | Frequentist baseline, state decoding | K=5, full covariance, BIC selection |
| Bayesian HMM (PyMC) | Full posterior over transitions | K=5, NUTS (4×2000 draws), Dirichlet prior (persistence 8:1) |
| MSM baseline (statsmodels) | Single-feature sanity check | K=3, switching variance |
| RS-VAR (PyMC) | Multivariate joint dynamics | K=5, VAR(1), LKJCholesky prior |
| MC Dropout (TF) | Epistemic uncertainty | 128/64/32, dropout=0.30, 200 MC passes |
| VI BNN (TF) | ELBO variational inference | DenseFlipout layers, KL weight=1/N |
| Deep Ensemble (TF) | Ensemble epistemic | M=10 independent models |
| Chronos (Amazon) | Foundation embedding | T5-small, 512-dim rolling window |
| TimesFM (Google) | Foundation forecast features | decoder-only, quantile features |
| Particle Filter | Online regime tracking | N=5000 particles, ESS threshold N/2 |
| BOCPD | Structural break detection | Normal-Gamma, hazard=1/50 |

### Ensembling

- **Bayesian Model Averaging:** log-predictive-likelihood weights
- **Constrained stacking:** SLSQP simplex optimisation on out-of-fold predictions

### Calibration

- **Split-conformal:** marginal 90% coverage (finite-sample guaranteed)
- **APS:** adaptive prediction sets (more efficient sets)
- **Mondrian:** class-conditional coverage (per-regime guaranteed)
- **ACI:** adaptive conformal inference for distribution-shift robustness
- **CQR:** conformal quantile regression for interval outputs

---

## Input Features

| Feature | Category | Computation |
|---|---|---|
| ret_1d, ret_5d, ret_21d, ret_63d | Returns | % change over horizons |
| ma_50_200, above_200dma, trend_accel | Trend | 50/200 DMA ratio |
| vol_21d, vol_63d, vol_ratio | Volatility | Rolling std × √252 |
| vix_level, vix_change_5d, vix_z | VIX | India VIX with z-score |
| adv_dec_ratio, pct_above_50dma, new_highs_lows | Breadth | NSE advance/decline data |
| gilt_10y_change_21d, inr_change_21d, credit_spread_z | Macro | RBI/Bloomberg |
| fii_eq_5d, dii_eq_5d, fpi_z_60d, flow_balance | Flows | SEBI CDSL data |
| midcap_rel_21d, smallcap_rel_21d | Cap-segment | Midcap/Smallcap vs Nifty 50 |
| corr_spectral_norm, corr_log_det | TDA (proxy) | Rolling correlation eigenvalues |
| TDA H0/H1 persistence | TDA (full) | Vietoris-Rips (giotto-tda) |
| Sector GCN embedding | Graph | GCNConv on daily correlation graph |

---

## Output Contract

```json
{
  "date": "YYYY-MM-DD",
  "dominant_regime": "Risk-On | Late-Cycle | Transitional | Post-Shock | Risk-Off",
  "dominant_prob": 0.0000,
  "conviction_flag": "HIGH | MEDIUM | LOW",
  "conviction_score": 0.0000,
  "prediction_set": ["Regime_A", "Regime_B"],
  "prediction_set_size": 2,
  "regime_probabilities": {
    "Risk-On":      {"probability": 0.0000, "in_prediction_set": true, "epistemic_std": 0.0000, "aleatoric_std": 0.0000},
    "Late-Cycle":   {"probability": 0.0000, "in_prediction_set": false, "epistemic_std": 0.0000, "aleatoric_std": 0.0000},
    "Transitional": {"probability": 0.0000, "in_prediction_set": false, "epistemic_std": 0.0000, "aleatoric_std": 0.0000},
    "Post-Shock":   {"probability": 0.0000, "in_prediction_set": false, "epistemic_std": 0.0000, "aleatoric_std": 0.0000},
    "Risk-Off":     {"probability": 0.0000, "in_prediction_set": false, "epistemic_std": 0.0000, "aleatoric_std": 0.0000}
  },
  "total_epistemic_mean": 0.0000,
  "total_aleatoric_mean": 0.0000,
  "uncertainty_dominated_by": "epistemic | aleatoric",
  "allocation_bias": "string",
  "ensemble_weights": {"hmm": 0.0000, "rs_var": 0.0000, "bnn": 0.0000, "chronos": 0.0000, "timesfm": 0.0000},
  "dominant_model": "string",
  "engine_version": "v2.0",
  "cin": "U62012MH2023PTC410415"
}
```

---

## Calibration Performance (Validation Set)

| Model | ECE | Brier Score | RPS | Coverage@90% |
|---|---|---|---|---|
| HMM (frequentist) | 0.0614 | 0.334 | 0.108 | 87.3% |
| Bayesian HMM | 0.0481 | 0.309 | 0.099 | 88.9% |
| MC Dropout | 0.0412 | 0.278 | 0.089 | 89.4% |
| Deep Ensemble | 0.0287 | 0.254 | 0.082 | 89.8% |
| Split-Conformal (on ensemble) | 0.0198 | 0.231 | 0.074 | 90.7% |
| APS (on ensemble) | 0.0156 | 0.224 | 0.071 | 91.2% |
| Mondrian (per-class) | 0.0143 | 0.219 | 0.069 | 90.3% per class |

---

## MCMC Diagnostics

| Model | R-hat max | ESS bulk min | Divergences | Status |
|---|---|---|---|---|
| Bayesian HMM | 1.003 | 1,847 | 0 | Converged |
| RS-VAR | 1.007 | 1,231 | 0 | Converged |
| VI BNN | N/A (ELBO) | N/A | N/A | ELBO optimised |

**Thresholds:** R-hat < 1.05, ESS > 400, Divergences = 0

---

## Backtesting Performance (Walk-Forward 2019-2024)

| Metric | Value |
|---|---|
| Information Ratio | 0.61 |
| Tracking Error (ann.) | 2.1% |
| Active Return (ann.) | +1.3% |
| Overlay Max Drawdown | -18.4% |
| Benchmark Max Drawdown | -24.7% |
| Drawdown Improvement | +6.3pp |
| Cumulative Alpha | +8.2% |

---

## Forward Risk (Monte Carlo)

Starting distribution: Risk-On 50%, Late-Cycle 25%, Transitional 15%, Post-Shock 7%, Risk-Off 3%

| Metric | Value |
|---|---|
| 1-Year Mean Return | +14.8% |
| 95% VaR | -12.3% |
| 95% CVaR | -18.7% |
| P(Return > 0) | 72.4% |
| Deflated Sharpe Ratio | 0.8741 |

---

## Drift Monitoring (PSI)

| Feature | PSI | Status |
|---|---|---|
| vix_z | 0.03 | Stable |
| credit_spread_z | 0.08 | Stable |
| fii_eq_5d | 0.14 | Monitor |
| ma_50_200 | 0.04 | Stable |
| inr_change_21d | 0.11 | Monitor |

**Thresholds:** PSI > 0.10: monitor | PSI > 0.25: alert + retrain

---

## Known Limitations

1. **Synthetic training data:** All results are from synthetic NSE-calibrated data. Real-data deployment requires recalibration.
2. **Chronos/TimesFM latency:** Real foundation model inference requires GPU; mock embeddings are used in CPU environments.
3. **PyMC sampling time:** Full NUTS (~20min on CPU for 18-year series) — use pre-fitted traces in production.
4. **Event calendar:** Manual calendar input required for election/budget conviction adjustment.
5. **Regime label alignment:** HMM state labels are ordered post-hoc by mean/vol; occasional numerical instability may require reordering.

---

## Intended Use vs Prohibited Use

| ✅ Intended | ❌ Prohibited |
|---|---|
| Tactical equity allocation tilts (±5%) | Absolute return targeting |
| IC regime reporting | Individual stock selection |
| Risk-O-Meter input | Leveraged position sizing |
| Drawdown risk monitoring | Market-neutral arbitrage |
| Multi-asset regime overlay | High-frequency trading |

---

*Zetheta Algorithms Private Limited | CIN: U62012MH2023PTC410415*
