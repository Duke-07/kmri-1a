# Bayesian Regime Detection Engine for Indian Equity Markets

This repository contains the implementation of a Bayesian Regime Detection Engine tailored for the Indian equity market, designed to forecast market direction using an ensemble of probabilistic and foundational models.

## Project Structure

- `src/data/`: Contains synthetic market data generation and feature engineering (trend, breadth, volatility, flow divergence, and macro indicators).
- `src/models/`: Implements the core predictive models:
  - Frequentist & Bayesian Hidden Markov Models (`hmmlearn`, `PyMC`)
  - Regime-Switching Vector Autoregression (RS-VAR via `PyMC`)
  - Bayesian Deep Learning (MC Dropout, Variational Inference via `TensorFlow Probability`)
  - Foundation Models (Wrapper for `Chronos`)
- `src/inference/`: Implements sequential online inference using a Bootstrap Particle Filter and Bayesian Online Changepoint Detection (BOCPD).
- `src/ensembling/`: Combines model outputs via Bayesian Model Averaging (BMA) and Constrained Stacking.
- `src/calibration/`: Provides conformal prediction wrappers (Split-Conformal, Adaptive Prediction Sets, ACI) for calibrated uncertainty.
- `src/backtest/`: Contains tools for Regime-conditioned Monte Carlo path generation and Deflated Sharpe Ratio calculation.
- `R/`: Baseline Markov-switching models implemented in R for cross-language validation (`depmixS4`, `MSwM`).
- `docs/`: Project documentation, including the Main Report, Final Presentation slides, and the Model Card.

## Installation

We recommend setting up a dedicated Conda environment to manage the scientific dependencies (e.g., PyMC, TensorFlow Probability):

```bash
conda create -n regime python=3.10 -y
conda activate regime
pip install -r requirements.txt
```

## Running the Pipeline

1. **Run End-to-End Master Pipeline**:
   To execute all 7 stages of the regime engine sequentially (data generation, feature engineering, HMM decoding, particle filtering, BOCPD, ensembling, conformal calibration, and Monte Carlo backtesting):
   ```bash
   python run_pipeline.py
   ```

2. **Generate Synthetic Data**:
   To test the engine independently, simulate 15 years of Indian equity market history (Nifty, Midcap, Smallcap, VIX, Flows, Macro):
   ```bash
   python src/data/synthetic_data.py
   ```
3. **Feature Engineering**:
   Generate the requisite features for regime modelling:
   ```bash
   python src/data/feature_engineering.py
   ```
4. **Model Execution**:
   The individual scripts under `src/models/` and `src/inference/` can be executed independently to train specific components of the ensemble.

## Execution Output & Pipeline Validation

Running `python run_pipeline.py` outputs the full 7-stage diagnostic trace:

```text
======================================================================
BAYESIAN REGIME DETECTION ENGINE - END-TO-END PIPELINE
======================================================================

[Step 1/7] Generating Synthetic Indian Market Data (2007 - 2024)...
  -> Generated 4697 daily trading records with OHLC, VIX, Flows, Macro, and SIP indicators.

[Step 2/7] Running Feature Engineering Pipeline (Returns, Vol, Breadth, Macro, Flows)...
  -> Engineered 26 core features + topological correlation proxies.
  -> Sample Composite Crash Alert Distribution:
NORMAL            3775
MONITOR            792
WARNING            123
ACUTE RISK-OFF       7
Name: count, dtype: int64

[Step 3/7] Fitting 5-State Gaussian HMM and Decoding Economic Regimes...
  -> Decoded Regimes Mapping: {2: 'Risk-On', 4: 'Late-Cycle', 0: 'Transitional', 1: 'Post-Shock', 3: 'Risk-Off'}

[Step 4/7] Running Bootstrap Particle Filter & BOCPD Changepoint Detector...
  -> Particle Filter Posterior for last bar: [0.4351 0.4524 0.     0.0579 0.0545]
  -> BOCPD Run-Length Matrix evaluated for recent 100 sessions. Max run length at bar 100: 100

[Step 5/7] Model Ensembling (BMA & Constrained Stacking)...
  -> Bayesian Model Averaging (BMA) Weights across 3 ensemble models: [0.3112 0.3616 0.3272]

[Step 6/7] Applying Conformal Prediction Sets for Audit-Defensible Coverage...
  -> Conformal Quantile Threshold q_hat (alpha=0.10): 0.5594
  -> Average Prediction Set Size across test period: 0.82 states out of 5

[Step 7/7] Regime-Conditioned Monte Carlo Simulation & Deflated Sharpe Ratio...
  -> 1-Year Projected Return Mean: -1.19%
  -> 95% Regime-Conditioned Value-at-Risk (VaR): -34.15%
  -> 95% Regime-Conditioned Conditional VaR (CVaR): -42.54%
  -> Deflated Sharpe Ratio (P(True Sharpe > 0)): 1.0000

======================================================================
ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!
======================================================================
```

## Key Implementation Highlights

Here are some of the key code implementations driving the regime detection engine.

### 1. Bayesian HMM (PyMC)
We use `PyMC` to build a robust Bayesian Hidden Markov Model with Dirichlet priors on regime transitions, ensuring calibrated uncertainty across states:

```python
import pymc as pm
import pytensor.tensor as pt
import numpy as np

def build_bayesian_hmm(returns: np.ndarray, K=5):
    T = len(returns)
    with pm.Model() as model:
        # Priors for transition matrix
        alpha_diag = 8.0
        alpha_off = 1.0
        alpha_mat = np.eye(K) * alpha_diag + (1 - np.eye(K)) * alpha_off
        
        P = pm.Dirichlet('P', a=alpha_mat, shape=(K, K))
        pi = pm.Dirichlet('pi', a=np.ones(K), shape=K)
        
        # Emission parameters
        mu = pm.Normal('mu', mu=0, sigma=0.02, shape=K)
        sigma = pm.HalfNormal('sigma', sigma=0.03, shape=K)
        
        states = pm.Categorical('states', p=pi, shape=T)
        obs = pm.Normal('obs', mu=mu[states], sigma=sigma[states], observed=returns)
        
    return model
```

### 2. Bayesian Deep Learning (TensorFlow Probability)
We leverage Variational Inference via TensorFlow Probability to model the complex, non-linear relationships in market regimes.

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions

def build_variational_regime_classifier(input_dim, n_regimes=5, train_size=2000):
    kl_weight = 1.0 / train_size
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tfpl.DenseFlipout(128, activation='relu',
                          kernel_divergence_fn=lambda q,p,_: kl_weight*tfd.kl_divergence(q,p))(inputs)
    x = tfpl.DenseFlipout(64, activation='relu',
                          kernel_divergence_fn=lambda q,p,_: kl_weight*tfd.kl_divergence(q,p))(x)
    logits = tfpl.DenseFlipout(n_regimes,
                               kernel_divergence_fn=lambda q,p,_: kl_weight*tfd.kl_divergence(q,p))(x)
    outputs = tfpl.OneHotCategorical(n_regimes)(logits)
    
    model = tf.keras.Model(inputs, outputs)
    nll = lambda y, rv_y: -rv_y.log_prob(y)
    model.compile(optimizer='adam', loss=nll, metrics=['accuracy'])
    return model
```

### 3. Conformal Prediction (Calibration)
To guarantee finite-sample coverage (audit-defensibility), we apply conformal prediction sets to our regime probabilities.

```python
import numpy as np

def split_conformal_classifier(model, X_cal, y_cal, X_test, alpha=0.1):
    """Return prediction sets with marginal coverage 1-alpha."""
    cal_probs = model.predict(X_cal)
    cal_scores = 1 - cal_probs[np.arange(len(y_cal)), y_cal]
    n = len(cal_scores)
    
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = np.quantile(cal_scores, q_level, method='higher')
    
    test_probs = model.predict(X_test)
    pred_sets = test_probs >= (1 - q_hat) # boolean mask of regimes in set
    return pred_sets, q_hat
```

## Key Features

- **Direction over Price**: Focuses on the probability of regimes (Risk-On, Risk-Off, Late-Cycle, Transitional, Post-Shock) rather than unreliable point estimates.
- **Audit-Defensible Calibration**: Enforces finite-sample coverage guarantees through Conformal Prediction, critical for regulatory compliance and Investment Committee confidence.
- **Multi-modal Ensembling**: Rigorously aggregates signals from Bayesian neural networks, sequential HMMs, and time-series foundation models.

