# Model Card: Bayesian Regime Detection Ensemble

## 1. Model Details
- **Name:** Bayesian Regime Detection Ensemble (v1.0)
- **Type:** Ensemble of HMM, RS-VAR, Bayesian Deep Learning, and Foundation Models.
- **Task:** Classify the Indian equity market into 5 discrete states.
- **Output:** Calibrated probability vector, conformal prediction set, epistemic/aleatoric uncertainty.

## 2. Intended Use
- **Primary Use:** Tactical allocation tilts for long-only mutual fund schemes.
- **Users:** Multi-Asset Solutions desk, Investment Committee.
- **Out of Scope:** Intraday high-frequency trading, derivative pricing.

## 3. Data & Features
- **Inputs:** Daily OHLC for Nifty 50, Midcap 100, Smallcap 100.
- **Macro/Flows:** India VIX, USD/INR, 10Y Gilt, AAA spread, FII/DII equity flows, monthly SIP inflows.
- **Feature Pipeline:** 30+ engineered features ensuring point-in-time constraints (no look-ahead bias).

## 4. Calibration and Validation
- **Conformal Wrapper:** Adaptive Prediction Sets (APS) configured for 90% marginal coverage.
- **Metrics Evaluated:** Expected Calibration Error (ECE), Brier Score, Ranked Probability Score (RPS).
- **Diagnostics:** Gelman-Rubin (R-hat < 1.01) for MCMC models; Population Stability Index (PSI) for input drift.

## 5. Ethical & Regulatory Considerations
- **SEBI Compliance:** Generates audit-defensible lineage for every daily prediction. Output probabilities map to Risk-O-Meter inputs and stewardship reporting.
- **Bias:** Uses survivorship-bias free data (accounting for delisted equities).

## 6. Maintenance & Drift Detection
- **Retraining Triggers:** 
  - PSI breach > 0.25 on core features.
  - 30-day rolling conformal coverage drops below 85%.
  - BOCPD triggers a structural changepoint.
