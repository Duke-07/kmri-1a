"""
Model Ensembling — BMA, Constrained Stacking, WAIC/LOO, Output Contract
=========================================================================
Implements Section A10 (all subsections):
  - Section A10.1: Bayesian Model Averaging (log predictive likelihood weights)
  - Section A10.2: Constrained stacking (simplex-constrained cross-entropy min)
  - Section A10.3: WAIC / PSIS-LOO model selection via ArviZ
  - Section A10.4: Combined regime output contract (structured daily output)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional

try:
    import arviz as az
    ARVIZ_AVAILABLE = True
except ImportError:
    az = None
    ARVIZ_AVAILABLE = False


# =============================================================================
# Section A10.1 — Bayesian Model Averaging
# =============================================================================

def bma_weights(log_predictive_likelihoods: np.ndarray) -> np.ndarray:
    """
    Compute BMA weights from out-of-sample log-predictive likelihoods.

    w_m = exp(l_m - max(l)) / sum_m exp(l_m - max(l))

    Parameters
    ----------
    log_predictive_likelihoods : (M,) — one scalar log-lik per model

    Returns
    -------
    np.ndarray (M,) — normalised BMA weights (sum to 1)
    """
    l = log_predictive_likelihoods - log_predictive_likelihoods.max()
    w = np.exp(l)
    return w / w.sum()


def bma_combine(model_probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Combine regime probability vectors via BMA.

    Parameters
    ----------
    model_probs : (M, K) — per-model regime probability vectors for one day
                  OR (M, N, K) for multiple days
    weights     : (M,) — BMA weights

    Returns
    -------
    np.ndarray (K,) or (N, K)
    """
    if model_probs.ndim == 2:
        return np.tensordot(weights, model_probs, axes=(0, 0))
    elif model_probs.ndim == 3:
        return np.einsum("m,mnk->nk", weights, model_probs)
    else:
        raise ValueError("model_probs must be (M,K) or (M,N,K)")


# =============================================================================
# Section A10.2 — Constrained Stacking
# =============================================================================

def fit_stacking_weights(
    base_probs: np.ndarray,
    y_true: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Fit stacking weights via simplex-constrained cross-entropy minimisation.

    base_probs : (M, N, K) — out-of-fold regime probabilities
    y_true     : (N,)      — integer true labels

    Returns
    -------
    np.ndarray (M,) — stacking weights (sum to 1, non-negative)
    """
    M, N, K = base_probs.shape
    onehot = np.eye(K)[y_true]  # (N, K)

    def neg_loglik(w: np.ndarray) -> float:
        w_clip = np.clip(w, 0, None)
        w_norm = w_clip / (w_clip.sum() + 1e-12)
        combined = np.einsum("m,mnk->nk", w_norm, base_probs)  # (N, K)
        combined = np.clip(combined, 1e-9, 1.0)
        return float(-np.mean(np.sum(onehot * np.log(combined), axis=1)))

    def grad(w: np.ndarray) -> np.ndarray:
        """Finite difference gradient (exact for small M)."""
        g = np.zeros_like(w)
        eps = 1e-5
        f0 = neg_loglik(w)
        for i in range(len(w)):
            w_p = w.copy(); w_p[i] += eps
            g[i] = (neg_loglik(w_p) - f0) / eps
        return g

    w0   = np.full(M, 1.0 / M)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    bnds = [(0.0, 1.0)] * M

    res  = minimize(neg_loglik, w0, bounds=bnds, constraints=cons,
                    method="SLSQP", tol=tol,
                    options={"maxiter": 500, "ftol": tol})

    weights = np.clip(res.x, 0, None)
    weights /= weights.sum()
    return weights


# =============================================================================
# Section A10.3 — WAIC / PSIS-LOO Model Comparison
# =============================================================================

def waic_loo_comparison(traces: dict) -> pd.DataFrame:
    """
    Compare Bayesian models by WAIC and PSIS-LOO using ArviZ.

    Parameters
    ----------
    traces : dict {model_name: arviz.InferenceData}

    Returns
    -------
    pd.DataFrame with: model, elpd_waic, elpd_loo, loo_se, n_pareto_k_bad
    """
    if not ARVIZ_AVAILABLE:
        return pd.DataFrame({"error": ["arviz not installed"]})

    rows = []
    for name, trace in traces.items():
        row = {"model": name}
        try:
            waic_res = az.waic(trace)
            row["elpd_waic"]    = round(float(waic_res.elpd_waic), 2)
            row["waic_se"]      = round(float(waic_res.se), 2)
        except Exception as e:
            row["elpd_waic"] = f"err: {e}"

        try:
            loo_res = az.loo(trace, pointwise=True)
            row["elpd_loo"]        = round(float(loo_res.elpd_loo), 2)
            row["loo_se"]          = round(float(loo_res.se), 2)
            row["n_pareto_k_bad"]  = int((loo_res.pareto_k.values > 0.7).sum())
        except Exception as e:
            row["elpd_loo"] = f"err: {e}"

        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    # Rank by LOO if available
    if "elpd_loo" in df.columns:
        try:
            df["loo_rank"] = df["elpd_loo"].rank(ascending=False).astype(int)
        except Exception:
            pass
    return df


# =============================================================================
# Section A10.4 — Combined Regime Output Contract
# =============================================================================

def build_regime_output(
    ensemble_probs: np.ndarray,
    conformal_set: np.ndarray,
    epistemic_uncertainty: np.ndarray,
    aleatoric_uncertainty: np.ndarray,
    model_weights: dict,
    conviction_threshold: float = 0.6,
    regime_names: Optional[list] = None,
    date: Optional[str] = None,
) -> dict:
    """
    Construct the standardised daily regime output contract.

    This structured output is the single endpoint consumed by the
    portfolio overlay engine, the Investment Committee artefact
    generator, and the regulatory reporting layer.

    Parameters
    ----------
    ensemble_probs         : (K,) combined regime probability vector
    conformal_set          : (K,) boolean mask — regimes in the prediction set
    epistemic_uncertainty  : (K,) model uncertainty (disagreement)
    aleatoric_uncertainty  : (K,) data uncertainty (irreducible noise)
    model_weights          : dict {model_name: weight}
    conviction_threshold   : scalar — min max_prob for high-conviction call
    regime_names           : list of K regime label strings

    Returns
    -------
    dict — full structured output with lineage metadata
    """
    if regime_names is None:
        regime_names = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]

    K = len(ensemble_probs)
    dominant_idx    = int(ensemble_probs.argmax())
    dominant_regime = regime_names[dominant_idx]
    max_prob        = float(ensemble_probs[dominant_idx])

    # Conviction: normalised by set size (smaller set = higher conviction)
    set_size        = int(conformal_set.sum())
    conviction      = float(np.clip(max_prob * (1 - epistemic_uncertainty[dominant_idx]), 0, 1))
    conviction_flag = "HIGH" if max_prob >= conviction_threshold else (
                       "MEDIUM" if max_prob >= 0.40 else "LOW"
                    )

    # Per-regime breakdown
    regime_breakdown = {}
    for k, name in enumerate(regime_names[:K]):
        regime_breakdown[name] = {
            "probability":         round(float(ensemble_probs[k]), 4),
            "in_prediction_set":   bool(conformal_set[k]),
            "epistemic_std":       round(float(epistemic_uncertainty[k]), 4),
            "aleatoric_std":       round(float(aleatoric_uncertainty[k]), 4),
        }

    output = {
        # ── Primary Output ───────────────────────────────────────────────────
        "date":                   date or pd.Timestamp.today().strftime("%Y-%m-%d"),
        "dominant_regime":        dominant_regime,
        "dominant_prob":          round(max_prob, 4),
        "conviction_flag":        conviction_flag,
        "conviction_score":       round(conviction, 4),
        "prediction_set":         [regime_names[k] for k in range(K) if conformal_set[k]],
        "prediction_set_size":    set_size,
        # ── Uncertainty Budget ───────────────────────────────────────────────
        "total_epistemic_mean":   round(float(epistemic_uncertainty.mean()), 4),
        "total_aleatoric_mean":   round(float(aleatoric_uncertainty.mean()), 4),
        "uncertainty_dominated_by": (
            "epistemic" if epistemic_uncertainty.mean() > aleatoric_uncertainty.mean()
            else "aleatoric"
        ),
        # ── Full Regime Breakdown ────────────────────────────────────────────
        "regime_probabilities":   regime_breakdown,
        # ── Ensemble Lineage ─────────────────────────────────────────────────
        "ensemble_weights":       {k: round(v, 4) for k, v in model_weights.items()},
        "dominant_model":         max(model_weights, key=model_weights.get),
        # ── Portfolio Action Guidance ────────────────────────────────────────
        "allocation_bias": _allocation_bias(dominant_regime, conviction_flag),
        # ── Audit Metadata ───────────────────────────────────────────────────
        "engine_version":         "v2.0",
        "model_stack":            list(model_weights.keys()),
    }

    return output


def _allocation_bias(regime: str, conviction: str) -> str:
    """Map dominant regime + conviction to an allocation bias description."""
    bias_map = {
        ("Risk-On",      "HIGH"):   "Tilt toward equity beta; reduce cash buffer",
        ("Risk-On",      "MEDIUM"): "Modest equity overweight; maintain core positions",
        ("Risk-On",      "LOW"):    "Monitor; no tactical action recommended",
        ("Late-Cycle",   "HIGH"):   "Rotate to quality; reduce duration",
        ("Late-Cycle",   "MEDIUM"): "Begin defensive tilt; monitor breadth",
        ("Transitional", "HIGH"):   "Flat / balanced; await resolution",
        ("Transitional", "MEDIUM"): "Reduce concentration; hold cash",
        ("Transitional", "LOW"):    "Maintain positions; high uncertainty",
        ("Post-Shock",   "HIGH"):   "Mean-reversion opportunity; selective re-entry",
        ("Post-Shock",   "MEDIUM"): "Cautious re-entry; await confirming breadth",
        ("Risk-Off",     "HIGH"):   "Raise cash; rotate to defensives and debt",
        ("Risk-Off",     "MEDIUM"): "Defensive tilt; reduce equity beta",
        ("Risk-Off",     "LOW"):    "Hold current defensive positions",
    }
    return bias_map.get((regime, conviction), "No specific action — await higher conviction")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    M, N, K = 3, 100, 5

    m1_probs = np.random.dirichlet(np.ones(K), size=N)
    m2_probs = np.random.dirichlet(np.ones(K), size=N)
    m3_probs = np.random.dirichlet(np.ones(K), size=N)
    base_probs = np.stack([m1_probs, m2_probs, m3_probs])  # (3, 100, 5)
    y_true = np.random.randint(0, K, size=N)

    # BMA
    log_liks   = np.array([-1.10, -0.92, -1.05])
    bma_w      = bma_weights(log_liks)
    bma_probs  = bma_combine(base_probs, bma_w)
    print(f"BMA Weights: {np.round(bma_w, 4)}")
    print(f"BMA Combined probs (first bar): {np.round(bma_probs[0], 4)}")

    # Stacking
    stack_w = fit_stacking_weights(base_probs, y_true)
    print(f"\nStacking Weights: {np.round(stack_w, 4)}")

    # Output contract
    ep = np.random.uniform(0.01, 0.05, K)
    al = np.random.uniform(0.05, 0.10, K)
    cs = np.array([True, False, True, False, False])
    model_weights = {"hmm": 0.35, "rs_var": 0.30, "bnn": 0.20, "chronos": 0.15}
    output = build_regime_output(bma_probs[0], cs, ep, al, model_weights, date="2024-01-15")
    print("\n=== Combined Regime Output Contract ===")
    for k, v in output.items():
        print(f"  {k}: {v}")
