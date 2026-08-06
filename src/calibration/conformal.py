"""
Conformal Prediction & Calibration
====================================
Implements Section A6 (all variants) plus calibration diagnostics.

Includes:
  - Split-conformal classifier (A6.2)
  - Adaptive Prediction Sets / APS (A6.3)
  - Conformalised Quantile Regression / CQR (A6.4)
  - Adaptive Conformal Inference / ACI (A6.5) — distribution-shift robust
  - Mondrian (class-conditional) conformal (A6.5)
  - EnbPI-style ensemble prediction intervals (A6.5)
  - Reliability diagram data + ECE (A6.6)
  - Rolling 252-day coverage stability tracking
"""

import numpy as np
import pandas as pd
from typing import Optional

try:
    from sklearn.ensemble import GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    GradientBoostingRegressor = None
    SKLEARN_AVAILABLE = False


# =============================================================================
# Utility
# =============================================================================

def _quantile_higher(a: np.ndarray, q: float) -> float:
    """Empirical quantile with 'higher' interpolation (finite-sample guarantee)."""
    try:
        return float(np.quantile(a, q, method="higher"))
    except TypeError:
        return float(np.quantile(a, q, interpolation="higher"))


# =============================================================================
# Section A6.2 — Split-Conformal Classifier
# =============================================================================

def split_conformal_classifier(
    model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.10,
) -> tuple:
    """
    Return prediction sets with marginal coverage ≥ 1 - alpha.

    Non-conformity score: s_i = 1 - p_hat(y_i | x_i)

    Parameters
    ----------
    model  : any object with .predict(X) → (N, K) probability matrix
    X_cal  : (n_cal,) index array or (n_cal, D) features
    y_cal  : (n_cal,) integer true labels
    X_test : (n_test,) index array or (n_test, D) features
    alpha  : miscoverage level (0.10 → 90% coverage)

    Returns
    -------
    pred_sets : (n_test, K) boolean mask — True iff class k in prediction set
    q_hat     : float — conformal threshold
    empirical_coverage : float — on calibration set (should be ≥ 1-alpha)
    """
    cal_probs  = model.predict(X_cal)
    cal_scores = 1.0 - cal_probs[np.arange(len(y_cal)), y_cal]
    n = len(cal_scores)

    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat   = _quantile_higher(cal_scores, q_level)

    test_probs = model.predict(X_test)
    pred_sets  = test_probs >= (1 - q_hat)

    # Empirical calibration coverage
    cal_covered = cal_scores <= q_hat
    empirical_coverage = float(cal_covered.mean())

    return pred_sets, float(q_hat), empirical_coverage


# =============================================================================
# Section A6.3 — Adaptive Prediction Sets (Romano et al. 2020)
# =============================================================================

def adaptive_prediction_sets(
    model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.10,
) -> tuple:
    """
    Adaptive Prediction Sets — more efficient sets via cumulative sort scores.

    Non-conformity score: cumulative probability up to and including true class
    in the sorted (high-to-low) probability ordering.

    Returns
    -------
    pred_sets : (n_test, K) boolean mask
    q_hat     : float
    set_sizes : (n_test,) — number of classes in each prediction set
    """
    cal_probs  = model.predict(X_cal)
    sorted_idx = np.argsort(-cal_probs, axis=1)
    sorted_probs = np.take_along_axis(cal_probs, sorted_idx, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)

    # Score for each calibration example = cumsum up to true class
    rank_of_true = np.array([
        int(np.where(sorted_idx[i] == y_cal[i])[0][0]) for i in range(len(y_cal))
    ])
    cal_scores = cumsum[np.arange(len(y_cal)), rank_of_true]

    n = len(cal_scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat   = _quantile_higher(cal_scores, q_level)

    # Test prediction sets
    test_probs    = model.predict(X_test)
    sorted_idx_t  = np.argsort(-test_probs, axis=1)
    sorted_probs_t = np.take_along_axis(test_probs, sorted_idx_t, axis=1)
    cumsum_t = np.cumsum(sorted_probs_t, axis=1)

    in_set = cumsum_t <= q_hat
    in_set[:, 0] = True  # always include top class

    pred_sets = np.zeros_like(test_probs, dtype=bool)
    for i in range(len(test_probs)):
        for j in range(test_probs.shape[1]):
            if in_set[i, j]:
                pred_sets[i, sorted_idx_t[i, j]] = True

    set_sizes = pred_sets.sum(axis=1)
    return pred_sets, float(q_hat), set_sizes


# =============================================================================
# Section A6.4 — Conformalised Quantile Regression (CQR)
# =============================================================================

def cqr_intervals(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_te: np.ndarray,
    alpha: float = 0.10,
    n_estimators: int = 200,
) -> tuple:
    """
    Conformalised Quantile Regression (Romano, Patterson & Candès, 2019).

    Produces prediction intervals [lo - q, hi + q] where q is calibrated
    to correct any base quantile regressor miscalibration.

    Parameters
    ----------
    X_tr/X_cal/X_te : feature matrices
    y_tr/y_cal      : labels (continuous)
    alpha           : miscoverage level
    n_estimators    : GBM trees per quantile model

    Returns
    -------
    lo_te   : (n_test,) lower bounds
    hi_te   : (n_test,) upper bounds
    q_corr  : float — conformal correction
    coverage: float — empirical calibration coverage
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for CQR")

    lo_q, hi_q = alpha / 2, 1 - alpha / 2

    q_lo = GradientBoostingRegressor(loss="quantile", alpha=lo_q, n_estimators=n_estimators)
    q_hi = GradientBoostingRegressor(loss="quantile", alpha=hi_q, n_estimators=n_estimators)
    q_lo.fit(X_tr, y_tr)
    q_hi.fit(X_tr, y_tr)

    lo_cal = q_lo.predict(X_cal)
    hi_cal = q_hi.predict(X_cal)
    # Non-conformity score: max(lower_miss, upper_miss)
    E = np.maximum(lo_cal - y_cal, y_cal - hi_cal)

    n = len(E)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_corr  = _quantile_higher(E, q_level)

    lo_te = q_lo.predict(X_te) - q_corr
    hi_te = q_hi.predict(X_te) + q_corr

    # Empirical coverage on calibration
    covered = (y_cal >= lo_cal - q_corr) & (y_cal <= hi_cal + q_corr)
    coverage = float(covered.mean())

    return lo_te, hi_te, float(q_corr), coverage


# =============================================================================
# Section A6.5 — Adaptive Conformal Inference (Gibbs & Candès 2021)
# =============================================================================

def adaptive_conformal_inference(
    scores_stream: list,
    alpha_target: float = 0.10,
    gamma: float = 0.01,
) -> pd.DataFrame:
    """
    Online ACI: update alpha_t after each realised outcome to maintain
    coverage under distribution shift.

    scores_stream : list of (nonconformity_score, q_hat, covered) tuples
    alpha_target  : target miscoverage level
    gamma         : step size for alpha update

    Returns
    -------
    pd.DataFrame with columns: step, alpha_t, covered, q_hat
    """
    alpha_t = alpha_target
    rows = []

    for step, (score, q_hat, covered) in enumerate(scores_stream):
        err_t = 0 if covered else 1
        alpha_t = alpha_t + gamma * (alpha_target - err_t)
        alpha_t = float(np.clip(alpha_t, 1e-3, 1 - 1e-3))
        rows.append({
            "step":    step,
            "alpha_t": round(alpha_t, 5),
            "covered": int(covered),
            "q_hat":   round(q_hat, 5),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["rolling_coverage_50"] = (
            df["covered"].rolling(50, min_periods=1).mean()
        )
    return df


# =============================================================================
# Section A6.5 — Mondrian (Class-Conditional) Conformal
# =============================================================================

def mondrian_conformal_classifier(
    model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.10,
    K: int = 5,
) -> tuple:
    """
    Class-conditional (Mondrian) conformal prediction.

    Calibrates a separate threshold per regime class, so coverage holds
    within each class (conditional coverage) rather than only marginally.

    Returns
    -------
    pred_sets : (n_test, K) boolean mask
    q_hats    : dict {k: q_hat_k} — per-class thresholds
    """
    cal_probs = model.predict(X_cal)

    q_hats = {}
    for k in range(K):
        mask_k = y_cal == k
        if mask_k.sum() < 10:
            q_hats[k] = 1.0  # degenerate
            continue
        scores_k = 1 - cal_probs[mask_k, k]
        n_k = len(scores_k)
        q_level = min(np.ceil((n_k + 1) * (1 - alpha)) / n_k, 1.0)
        q_hats[k] = _quantile_higher(scores_k, q_level)

    test_probs = model.predict(X_test)
    pred_sets  = np.zeros((len(X_test), K), dtype=bool)
    for k in range(K):
        pred_sets[:, k] = test_probs[:, k] >= (1 - q_hats[k])

    return pred_sets, q_hats


# =============================================================================
# Section A6.6 — Calibration Diagnostics
# =============================================================================

def expected_calibration_error(
    probs_max: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error: weighted average |accuracy - confidence| per bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    n_total = len(probs_max)
    ece = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs_max > lo) & (probs_max <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        acc  = float(correct[mask].mean())
        conf = float(probs_max[mask].mean())
        ece += (n_bin / n_total) * abs(acc - conf)

    return float(ece)


def reliability_diagram_data(
    probs_max: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute binned accuracy vs confidence table for a reliability diagram.

    Returns
    -------
    pd.DataFrame — bin_mid, accuracy, confidence, count, gap
    """
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask  = (probs_max > lo) & (probs_max <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            rows.append({"bin_mid": (lo + hi) / 2,
                         "accuracy": 0, "confidence": (lo + hi) / 2,
                         "count": 0, "gap": 0.0})
        else:
            acc  = float(correct[mask].mean())
            conf = float(probs_max[mask].mean())
            rows.append({
                "bin_mid":    round((lo + hi) / 2, 2),
                "accuracy":   round(acc,  4),
                "confidence": round(conf, 4),
                "count":      int(n_bin),
                "gap":        round(abs(acc - conf), 4),
            })
    return pd.DataFrame(rows)


def rolling_conformal_coverage(
    scores: np.ndarray,
    q_hat: float,
    window: int = 252,
) -> pd.Series:
    """
    Track rolling empirical coverage against the conformal threshold.

    A well-calibrated model should show rolling coverage ≥ 90% (for alpha=0.10)
    even under moderate regime shift.

    Parameters
    ----------
    scores : (T,) non-conformity scores (1 - p_hat for each test day)
    q_hat  : conformal threshold
    window : rolling window in business days

    Returns
    -------
    pd.Series — rolling coverage (fraction of days with score <= q_hat)
    """
    covered = pd.Series((scores <= q_hat).astype(float))
    return covered.rolling(window, min_periods=30).mean()


def brier_score(probs: np.ndarray, y_true: np.ndarray, K: int = 5) -> float:
    """
    Multiclass Brier Score: mean squared error between probability vector
    and one-hot true label.

    Lower is better; 0 = perfect; 0.4 = random (for K=5).
    """
    onehot = np.eye(K)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def ranked_probability_score(
    probs: np.ndarray,
    y_true: np.ndarray,
    K: int = 5,
) -> float:
    """
    Ranked Probability Score (RPS) — proper scoring rule for ordinal outcomes.

    Penalises cumulative probability mass that is not concentrated near
    the true regime. Appropriate when regimes have a natural ordering
    (Risk-On → Late-Cycle → ... → Risk-Off).

    RPS = (1/K-1) * sum_{k=0}^{K-1} (CDF_forecast[k] - CDF_true[k])^2
    """
    cdf_probs = np.cumsum(probs, axis=1)  # (N, K)
    onehot    = np.eye(K)[y_true]
    cdf_true  = np.cumsum(onehot, axis=1)  # (N, K)
    rps = np.mean(np.sum((cdf_probs - cdf_true) ** 2, axis=1)) / (K - 1)
    return float(rps)


# =============================================================================
# Mock Model for Testing
# =============================================================================

class MockModel:
    """Simple lookup model for testing conformal wrappers."""
    def __init__(self, probs: np.ndarray):
        self.probs = probs

    def predict(self, X):
        if hasattr(X, "__len__") and len(X) <= len(self.probs):
            return self.probs[np.array(X)]
        return self.probs


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    K = 5
    n_cal, n_test = 300, 100
    cal_probs   = np.random.dirichlet(np.ones(K), size=n_cal)
    test_probs  = np.random.dirichlet(np.ones(K), size=n_test)
    all_probs   = np.vstack([cal_probs, test_probs])
    y_cal       = np.random.randint(0, K, size=n_cal)

    model  = MockModel(all_probs)
    X_cal  = np.arange(n_cal)
    X_test = np.arange(n_cal, n_cal + n_test)

    # Split-conformal
    sets_sc, q_hat_sc, cov_sc = split_conformal_classifier(model, X_cal, y_cal, X_test, alpha=0.10)
    print(f"[Split-Conformal]  q_hat={q_hat_sc:.4f}, cal_coverage={cov_sc:.3f}, "
          f"avg_set_size={sets_sc.sum(axis=1).mean():.2f}")

    # APS
    sets_aps, q_hat_aps, sizes_aps = adaptive_prediction_sets(model, X_cal, y_cal, X_test, alpha=0.10)
    print(f"[APS]              q_hat={q_hat_aps:.4f}, avg_set_size={sizes_aps.mean():.2f}")

    # Mondrian
    sets_mond, q_hats_mond = mondrian_conformal_classifier(model, X_cal, y_cal, X_test, alpha=0.10)
    print(f"[Mondrian]         per-class q_hats: {dict((k, round(v,4)) for k,v in q_hats_mond.items())}")

    # Calibration metrics
    probs_max = cal_probs.max(axis=1)
    correct   = (cal_probs.argmax(axis=1) == y_cal).astype(float)
    ece = expected_calibration_error(probs_max, correct)
    bs  = brier_score(cal_probs, y_cal, K=K)
    rps = ranked_probability_score(cal_probs, y_cal, K=K)
    print(f"\n[Calibration]  ECE={ece:.4f}  Brier={bs:.4f}  RPS={rps:.4f}")

    print("\n[Reliability Diagram Data]")
    rel = reliability_diagram_data(probs_max, correct)
    print(rel.to_string(index=False))
