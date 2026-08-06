"""
Single-Feature Markov-Switching Baseline — statsmodels
=======================================================
Implements the single-feature Markov-switching regression baseline (Section A8.2)
as the entry point before the full Bayesian RS-VAR.

Uses statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.
"""

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    STATSMODELS_AVAILABLE = True
except ImportError:
    MarkovRegression = None
    STATSMODELS_AVAILABLE = False
    print("[msm_baseline] statsmodels not available")


def fit_msm_regression(
    y: np.ndarray,
    k_regimes: int = 3,
    switching_variance: bool = True,
    switching_mean: bool = True,
    trend: str = "c",
) -> object:
    """
    Fit a Markov-switching mean-variance regression model.

    Parameters
    ----------
    y                  : 1D array of daily returns
    k_regimes          : number of regimes (typically 2 or 3)
    switching_variance : allow regime-conditional variance
    switching_mean     : allow regime-conditional intercept (mean)
    trend              : 'c' (constant), 'n' (no trend), 'ct'

    Returns
    -------
    Fitted MarkovRegressionResults object
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required: pip install statsmodels>=0.14")

    mod = MarkovRegression(
        y,
        k_regimes=k_regimes,
        trend=trend,
        switching_variance=switching_variance,
        switching_mean=switching_mean,
    )
    res = mod.fit(search_reps=10, search_scale=0.5)
    return res


def msm_regime_summary(res) -> pd.DataFrame:
    """
    Extract regime-conditional parameters and summary from a fitted MSM.

    Returns
    -------
    pd.DataFrame with per-regime mean and variance estimates
    """
    rows = []
    K = res.k_regimes
    for k in range(K):
        try:
            mean = float(res.params[f"[{k}]const"]) if f"[{k}]const" in res.params.index else 0.0
        except Exception:
            mean = 0.0
        try:
            var_key = [p for p in res.params.index if f"[{k}]sigma2" in p]
            var = float(res.params[var_key[0]]) if var_key else float(res.params["sigma2"])
        except Exception:
            var = np.nan
        rows.append({
            "Regime": k,
            "Const_Mean": round(mean, 6),
            "Variance": round(var, 8) if not np.isnan(var) else np.nan,
            "Ann_Vol": round(np.sqrt(var * 252), 4) if not np.isnan(var) else np.nan,
        })

    return pd.DataFrame(rows).set_index("Regime")


def msm_smoothed_probs(res) -> pd.DataFrame:
    """
    Extract smoothed marginal regime probabilities from a fitted MSM.

    Returns
    -------
    pd.DataFrame (T, K) — smoothed probability at each time step
    """
    probs = res.smoothed_marginal_probabilities
    if hasattr(probs, "values"):
        arr = probs.values
    else:
        arr = np.array(probs)

    K = arr.shape[1] if arr.ndim == 2 else 1
    cols = [f"regime_{k}" for k in range(K)]
    return pd.DataFrame(arr, columns=cols)


def compare_msm_bic(
    y: np.ndarray,
    k_values: tuple[int, ...] = (2, 3, 4),
) -> pd.DataFrame:
    """
    Compare Markov-switching models by BIC across different K.

    Returns
    -------
    pd.DataFrame with: K, LogLik, AIC, BIC
    """
    rows = []
    for k in k_values:
        try:
            res = fit_msm_regression(y, k_regimes=k)
            rows.append({
                "K": k,
                "LogLik": round(res.llf, 2),
                "AIC": round(res.aic, 2),
                "BIC": round(res.bic, 2),
            })
        except Exception as e:
            rows.append({"K": k, "LogLik": np.nan, "AIC": np.nan,
                         "BIC": np.nan, "Error": str(e)})

    df = pd.DataFrame(rows).set_index("K")
    if "BIC" in df.columns and not df["BIC"].isna().all():
        df["BIC_rank"] = df["BIC"].rank().astype(int)
    return df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.synthetic_data import generate_synthetic_market_data

    df, _ = generate_synthetic_market_data(seed=42)
    returns = df["Close"].pct_change().dropna().values

    print("=== Markov-Switching Model Comparison (K=2,3,4) ===")
    bic_table = compare_msm_bic(returns, k_values=(2, 3, 4))
    print(bic_table.to_string())

    print("\n=== Fitting 3-Regime MSM Baseline ===")
    res = fit_msm_regression(returns, k_regimes=3)
    print(res.summary().tables[0])

    print("\n=== Regime Parameter Summary ===")
    print(msm_regime_summary(res).to_string())

    print("\n=== Smoothed Regime Probabilities (last 10 days) ===")
    probs = msm_smoothed_probs(res)
    print(probs.tail(10).to_string())
