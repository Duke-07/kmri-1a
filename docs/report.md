# Bayesian Regime Detection Engine for Equity Direction Forecasting
**Final Project Report**

## Executive Summary
This report presents a next-generation Bayesian Regime Detection Engine designed to inform tactical allocation tilts for the Indian equity market. Overcoming the limitations of point-forecast models (signal-to-noise ratio, non-stationarity, and untestable uncertainty), this engine provides calibrated probabilities of market direction. It utilizes an ensemble of Hidden Markov Models, Bayesian deep learning, foundation models, and regime-switching vector autoregression (RS-VAR), outputting audit-defensible, conformalised predictions suited for regulatory and Investment Committee requirements.

## 1. Domain Fundamentals and Thesis
Traditional price prediction fails over mutual fund horizons due to unpredictable shocks and changing market regimes. The Directional Thesis advocates shifting from point-forecasts to probability distributions over market states (Risk-On, Risk-Off, Transitional, Late-Cycle, Post-Shock). 

## 2. Statistical Foundations & Ensembling
We employ a robust ensembling technique combining:
- **Frequentist & Bayesian HMMs:** For strong prior incorporation and calibrated uncertainty.
- **RS-VAR:** For multivariate dynamics (capturing cross-asset structure in crisis episodes).
- **Bayesian Deep Learning & Foundation Models:** (MC Dropout, Chronos, TimesFM) for high-dimensional feature abstraction.

These models are combined using Bayesian Model Averaging (BMA) and constrained stacking, ensuring that the engine dynamically weights the most accurate model based on predictive likelihood.

## 3. Conformal Prediction and Calibration
To guarantee finite-sample coverage, we apply Split-Conformal Classification and Adaptive Prediction Sets (APS). This translates the probability outputs into prediction sets with an exact marginal coverage (e.g., 90%), essential for institutional credibility.

## 4. Case Studies
### The 2020 COVID Regime Transition
- **Context:** Nifty fell 37.6% over a few weeks with extreme VIX spikes.
- **Model Behavior:** The RS-VAR and sequential BOCPD engine rapidly transitioned from Risk-On to Risk-Off as breadth collapsed and FII outflows accelerated, preserving capital relative to a buy-hold benchmark.

### The 2018 IL&FS Credit Shock
- **Context:** Mid-cap and small-cap indexes crashed while large caps remained stable.
- **Model Behavior:** By utilizing cap-segmented divergence features, the engine correctly identified a Transitional / Late-Cycle regime before the large-cap indices corrected.

## 5. Backtesting and Simulation
A walk-forward evaluation protocol with purged cross-validation was employed. Regime-conditioned Monte Carlo simulations indicate the probability-scaled allocation overlay significantly improves the Information Ratio and minimizes maximum drawdown without excessive turnover.

## Conclusion
The Bayesian Regime Detection Engine fulfills all regulatory and institutional requirements by providing calibrated, explainable, and reproducible regime probabilities for Indian equity markets.
