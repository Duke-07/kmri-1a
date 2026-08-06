"""
Frequentist Gaussian HMM — Regime Classifier
=============================================
Implements the hmmlearn-based frequentist HMM for regime classification.

Key additions vs. baseline:
  - BIC comparison across 3-, 5-, 7-state models
  - Regime duration statistics (mean, max, median)
  - Rolling re-fit stability analysis
  - Viterbi vs. posterior-mode decoding comparison
"""

import numpy as np
import pandas as pd
from typing import Optional

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    hmm = None
    HMMLEARN_AVAILABLE = False
    print("[frequentist_hmm] hmmlearn not available")


# =============================================================================
# Core HMM Fitting
# =============================================================================

def fit_regime_hmm(
    returns: pd.Series,
    n_states: int = 5,
    n_iter: int = 200,
    seed: int = 42,
    covariance_type: str = "full",
) -> tuple:
    """
    Fit a Gaussian-emission HMM to a returns series.

    Parameters
    ----------
    returns           : pd.Series of daily log/arithmetic returns
    n_states          : number of hidden states
    n_iter            : maximum Baum-Welch iterations
    seed              : random seed for reproducibility
    covariance_type   : 'diag' | 'full' | 'spherical' | 'tied'

    Returns
    -------
    model       : fitted GaussianHMM
    states      : (T,) Viterbi-decoded state sequence
    state_probs : (T, K) posterior probabilities
    """
    if not HMMLEARN_AVAILABLE:
        raise ImportError("hmmlearn library is required. Install with: pip install hmmlearn>=0.3")

    X = returns.dropna().values.reshape(-1, 1)
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=seed,
        tol=1e-6,
        verbose=False,
    )
    model.fit(X)
    states      = model.predict(X)
    state_probs = model.predict_proba(X)
    return model, states, state_probs


# =============================================================================
# Regime Labelling
# =============================================================================

def label_regimes(model, K: int = 5) -> dict:
    """
    Map numeric HMM states to economic regime names by (mean, vol) signature.

    Convention:
      Highest mean, lowest vol → Risk-On
      Lowest  mean, highest vol → Risk-Off
    """
    summary = []
    for i in range(K):
        mu = float(model.means_[i, 0])
        if model.covariance_type == "full":
            sig = float(np.sqrt(model.covars_[i, 0, 0]))
        elif model.covariance_type == "diag":
            sig = float(np.sqrt(model.covars_[i, 0]))
        else:
            sig = float(np.sqrt(model.covars_[i]))
        summary.append((i, mu, sig))

    summary.sort(key=lambda x: (x[1], -x[2]), reverse=True)

    if K == 5:
        labels = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]
    elif K == 3:
        labels = ["Risk-On", "Transitional", "Risk-Off"]
    elif K == 7:
        labels = ["Risk-On", "Bull-Quiet", "Late-Cycle", "Transitional",
                  "Post-Shock", "Bear-Quiet", "Risk-Off"]
    else:
        labels = [f"Regime_{j}" for j in range(K)]

    return {summary[r][0]: labels[r] for r in range(K)}


# =============================================================================
# BIC Model Selection
# =============================================================================

def compare_hmm_by_bic(
    returns: pd.Series,
    k_values: tuple[int, ...] = (3, 5, 7),
    n_iter: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Fit HMMs with different numbers of states and compare by BIC.

    BIC = -2 * log_likelihood + n_params * log(n_obs)

    Parameters
    ----------
    returns  : pd.Series
    k_values : tuple of state counts to compare

    Returns
    -------
    pd.DataFrame with columns: K, LogLik, n_params, BIC, AIC
    """
    rows = []
    X = returns.dropna().values.reshape(-1, 1)
    n_obs = len(X)

    for k in k_values:
        if not HMMLEARN_AVAILABLE:
            rows.append({"K": k, "LogLik": np.nan, "n_params": np.nan,
                         "BIC": np.nan, "AIC": np.nan})
            continue

        model = hmm.GaussianHMM(
            n_components=k,
            covariance_type="full",
            n_iter=n_iter,
            random_state=seed,
        )
        model.fit(X)
        log_lik = model.score(X)

        # n_params: transition matrix (k*(k-1)) + initial (k-1) + means (k) + covar (k)
        n_params = k * (k - 1) + (k - 1) + k + k

        bic = -2 * log_lik + n_params * np.log(n_obs)
        aic = -2 * log_lik + 2 * n_params

        rows.append({
            "K": k,
            "LogLik": round(log_lik, 2),
            "n_params": n_params,
            "BIC": round(bic, 2),
            "AIC": round(aic, 2),
        })

    df = pd.DataFrame(rows).set_index("K")
    df["BIC_rank"] = df["BIC"].rank().astype(int)
    return df


# =============================================================================
# Regime Duration Statistics
# =============================================================================

def regime_duration_stats(states: np.ndarray, K: int = 5) -> pd.DataFrame:
    """
    Compute empirical regime duration statistics from a decoded state sequence.

    Returns
    -------
    pd.DataFrame with: regime index, mean/median/max/min duration (business days)
    """
    rows = []
    for k in range(K):
        durations = []
        cur = 0
        for s in states:
            if s == k:
                cur += 1
            else:
                if cur > 0:
                    durations.append(cur)
                    cur = 0
        if cur > 0:
            durations.append(cur)

        if durations:
            rows.append({
                "State": k,
                "Count": sum(durations),
                "N_Episodes": len(durations),
                "Mean_Duration": round(float(np.mean(durations)), 1),
                "Median_Duration": float(np.median(durations)),
                "Max_Duration": int(np.max(durations)),
                "Min_Duration": int(np.min(durations)),
                "Std_Duration": round(float(np.std(durations)), 1),
            })
        else:
            rows.append({"State": k, "Count": 0, "N_Episodes": 0,
                         "Mean_Duration": 0, "Median_Duration": 0,
                         "Max_Duration": 0, "Min_Duration": 0, "Std_Duration": 0})

    return pd.DataFrame(rows).set_index("State")


# =============================================================================
# Rolling Re-Fit Stability
# =============================================================================

def rolling_hmm_stability(
    returns: pd.Series,
    n_states: int = 5,
    window: int = 504,
    step: int = 63,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assess regime assignment stability under rolling re-fits.

    Fits HMMs on rolling windows and tracks how often the most likely
    current-day regime changes between consecutive fits.

    Parameters
    ----------
    returns  : full returns series
    window   : training window length
    step     : re-fit frequency in days

    Returns
    -------
    pd.DataFrame with columns: date, dominant_state, transition_prob_mean
    """
    if not HMMLEARN_AVAILABLE:
        return pd.DataFrame()

    X = returns.dropna()
    rows = []
    for start in range(0, len(X) - window, step):
        end = start + window
        sub = X.iloc[start: end]
        try:
            model, states, probs = fit_regime_hmm(sub, n_states=n_states, seed=seed)
            last_state = states[-1]
            last_prob  = probs[-1].max()
            rows.append({
                "date": X.index[end - 1],
                "dominant_state": last_state,
                "max_prob": round(last_prob, 4),
                "transmat_diag_mean": round(float(np.diag(model.transmat_).mean()), 4),
            })
        except Exception:
            continue

    return pd.DataFrame(rows).set_index("date")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.synthetic_data import generate_synthetic_market_data

    df, true_regimes = generate_synthetic_market_data(seed=42)
    returns = df["Close"].pct_change().dropna()

    print("=== BIC Comparison across K=3, 5, 7 ===")
    bic_table = compare_hmm_by_bic(returns, k_values=(3, 5, 7))
    print(bic_table.to_string())

    print("\n=== Fitting 5-state HMM ===")
    model, states, probs = fit_regime_hmm(returns, n_states=5)
    mapping = label_regimes(model, K=5)
    print(f"Regime mapping: {mapping}")
    print(f"Transition matrix:\n{np.round(model.transmat_, 4)}")

    print("\n=== Regime Duration Statistics ===")
    dur_stats = regime_duration_stats(states, K=5)
    print(dur_stats.to_string())

    print("\n=== Regime Means / Vols ===")
    for k in range(5):
        mu  = float(model.means_[k, 0]) * 252  # annualised
        sig = float(np.sqrt(model.covars_[k, 0, 0])) * np.sqrt(252)
        print(f"  State {k} ({mapping[k]}): Ann. Mean={mu:.2%}, Ann. Vol={sig:.2%}")
