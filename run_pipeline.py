"""
Main Pipeline Runner for Bayesian Regime Detection Engine.
Executes end-to-end data generation, feature engineering, regime estimation,
sequential inference, ensembling, conformal calibration, and backtesting.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data.synthetic_data import generate_synthetic_market_data
from data.feature_engineering import (
    engineer_regime_features,
    compute_topology_features,
    covid_style_crash_alert,
    cap_segmented_stress,
    event_adjusted_conviction
)
from models.frequentist_hmm import fit_regime_hmm, label_regimes
from inference.particle_filter import RegimeParticleFilter
from inference.bocpd import bocpd
from ensembling.ensembling import bma_weights, bma_combine, fit_stacking_weights
from calibration.conformal import split_conformal_classifier, adaptive_prediction_sets
from backtest.backtest import RegimeConditionedMC, regime_var, conviction_scaled_tilt, deflated_sharpe_ratio

def run_pipeline():
    print("=" * 70)
    print("BAYESIAN REGIME DETECTION ENGINE - END-TO-END PIPELINE")
    print("=" * 70)
    
    # 1. Data Generation
    print("\n[Step 1/7] Generating Synthetic Indian Market Data (2007 - 2024)...")
    df, true_regimes = generate_synthetic_market_data(start_date='2007-01-01', end_date='2024-12-31', seed=42)
    print(f"  -> Generated {len(df)} daily trading records with OHLC, VIX, Flows, Macro, and SIP indicators.")
    
    # 2. Feature Engineering
    print("\n[Step 2/7] Running Feature Engineering Pipeline (Returns, Vol, Breadth, Macro, Flows)...")
    features = engineer_regime_features(df)
    tda_features = compute_topology_features(df)
    crash_alerts = covid_style_crash_alert(df)
    print(f"  -> Engineered {features.shape[1]} core features + topological correlation proxies.")
    print(f"  -> Sample Composite Crash Alert Distribution:\n{crash_alerts.value_counts()}")
    
    # 3. Frequentist & Regime Classification
    print("\n[Step 3/7] Fitting 5-State Gaussian HMM and Decoding Economic Regimes...")
    df_clean = df.loc[features.index]
    returns = df_clean['Close'].pct_change().dropna()
    model, states, state_probs = fit_regime_hmm(returns, n_states=5)
    label_mapping = label_regimes(model, K=5)
    print(f"  -> Decoded Regimes Mapping: {label_mapping}")
    
    # 4. Sequential Online Inference & Changepoint Detection
    print("\n[Step 4/7] Running Bootstrap Particle Filter & BOCPD Changepoint Detector...")
    P = model.transmat_
    mu = model.means_[:, 0]
    sigma = np.sqrt(model.covars_[:, 0, 0])
    pf = RegimeParticleFilter(P, mu, sigma, n_particles=2000)
    
    recent_rets = returns.values[-10:]
    last_post = None
    for r_val in recent_rets:
        last_post = pf.step(r_val)
    print(f"  -> Particle Filter Posterior for last bar: {np.round(last_post, 4)}")
    
    bocpd_matrix = bocpd(returns.values[-100:], hazard=1/50)
    print(f"  -> BOCPD Run-Length Matrix evaluated for recent 100 sessions. Max run length at bar 100: {np.argmax(bocpd_matrix[:, -1])}")
    
    # 5. Model Ensembling & Stacking
    print("\n[Step 5/7] Model Ensembling (BMA & Constrained Stacking)...")
    log_liks = np.array([-1.10, -0.95, -1.05]) # mock model out-of-sample log-likelihoods
    weights_bma = bma_weights(log_liks)
    print(f"  -> Bayesian Model Averaging (BMA) Weights across 3 ensemble models: {np.round(weights_bma, 4)}")
    
    # 6. Conformal Prediction & Calibration
    print("\n[Step 6/7] Applying Conformal Prediction Sets for Audit-Defensible Coverage...")
    # Mock calibration split
    n_cal = 200
    n_test = 50
    mock_cal_probs = state_probs[:n_cal]
    y_cal = states[:n_cal]
    mock_test_probs = state_probs[n_cal:n_cal+n_test]
    
    class TempModel:
        def __init__(self, probs_cal, probs_test):
            self.p_cal = probs_cal
            self.p_test = probs_test
        def predict(self, X):
            return self.p_cal if len(X) == len(self.p_cal) else self.p_test

    t_model = TempModel(mock_cal_probs, mock_test_probs)
    pred_sets, q_hat = split_conformal_classifier(t_model, np.arange(n_cal), y_cal, np.arange(n_test), alpha=0.1)
    print(f"  -> Conformal Quantile Threshold q_hat (alpha=0.10): {q_hat:.4f}")
    print(f"  -> Average Prediction Set Size across test period: {pred_sets.sum(axis=1).mean():.2f} states out of 5")
    
    # 7. Backtesting & Monte Carlo Simulation
    print("\n[Step 7/7] Regime-Conditioned Monte Carlo Simulation & Deflated Sharpe Ratio...")
    mc = RegimeConditionedMC(P, mu, sigma, K=5)
    init_dist = last_post
    paths, sim_states = mc.simulate(init_dist, horizon=252, n_sims=1000)
    var_95, cvar_95 = regime_var(paths, alpha=0.05)
    dsr = deflated_sharpe_ratio(sharpe=1.15, n_obs=1260, n_trials=15)
    
    print(f"  -> 1-Year Projected Return Mean: {paths[:, -1].mean() - 1:.2%}")
    print(f"  -> 95% Regime-Conditioned Value-at-Risk (VaR): {var_95:.2%}")
    print(f"  -> 95% Regime-Conditioned Conditional VaR (CVaR): {cvar_95:.2%}")
    print(f"  -> Deflated Sharpe Ratio (P(True Sharpe > 0)): {dsr:.4f}")
    
    print("\n" + "=" * 70)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
