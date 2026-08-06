"""
Bayesian Online Changepoint Detection (BOCPD)
=============================================
Implements the Normal-Gamma conjugate model for BOCPD (Adams & MacKay, 2007).

Additions vs baseline:
  - Validation against synthetic 2018/2020-equivalent breakpoints
  - Streaming Dirichlet/Beta posterior updates for online regime tracking
  - Online/batch reconciliation diagnostic
  - Probability-of-changepoint time series extraction
  - Two-speed design: nightly-batch + intraday-online reconciliation
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, beta as beta_dist


# =============================================================================
# Section A9.2 — Bayesian Online Changepoint Detection
# =============================================================================

def bocpd(
    data: np.ndarray,
    hazard: float = 1 / 100,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> np.ndarray:
    """
    Bayesian Online Changepoint Detection with Normal-Gamma model.

    Uses the Student-t predictive distribution (exact conjugate update
    for Normal data with unknown mean and variance).

    Parameters
    ----------
    data     : (T,) sequence of observations (returns)
    hazard   : P(changepoint at t) per step — 1/hazard_interval
    mu0, kappa0, alpha0, beta0 : Normal-Gamma hyperparameters

    Returns
    -------
    R : (T+1, T+1) run-length posterior matrix
        R[r, t] = P(run length = r at time t)
    """
    T = len(data)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0

    mu    = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta  = np.array([beta0])

    for t, x in enumerate(data):
        # Predictive: Student-t distribution under Normal-Gamma
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        df    = 2 * alpha
        # Student-t PDF  (numerically stable)
        pred = _student_t_pdf(x, df, mu, scale)

        # Growth probability (run continues)
        R[1: t + 2, t + 1] = R[0: t + 1, t] * pred * (1 - hazard)
        # Changepoint probability (run resets)
        R[0, t + 1] = np.sum(R[0: t + 1, t] * pred * hazard)

        # Normalise
        total = R[:, t + 1].sum()
        if total > 0:
            R[:, t + 1] /= total
        else:
            R[0, t + 1] = 1.0  # reset if numerical underflow

        # Update sufficient statistics (conjugate Normal-Gamma update)
        mu_new    = (kappa * mu + x) / (kappa + 1)
        kappa_new = kappa + 1
        alpha_new = alpha + 0.5
        beta_new  = beta + (kappa * (x - mu) ** 2) / (2 * (kappa + 1))

        mu    = np.concatenate([[mu0],    mu_new])
        kappa = np.concatenate([[kappa0], kappa_new])
        alpha = np.concatenate([[alpha0], alpha_new])
        beta  = np.concatenate([[beta0],  beta_new])

    return R


def _student_t_pdf(x: float, df: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Numerically stable Student-t PDF."""
    from scipy.special import gammaln
    z = (x - loc) / (scale + 1e-12)
    log_pdf = (
        gammaln((df + 1) / 2)
        - gammaln(df / 2)
        - 0.5 * np.log(df * np.pi)
        - np.log(scale + 1e-12)
        - ((df + 1) / 2) * np.log(1 + z ** 2 / df)
    )
    return np.exp(np.clip(log_pdf, -500, 0))


def changepoint_probability(R: np.ndarray) -> np.ndarray:
    """
    Extract P(changepoint at t) = R[0, t] for each time step.

    Returns
    -------
    np.ndarray (T,) — probability of changepoint at each time step
    """
    return R[0, 1:]  # R[0, t+1] = P(changepoint) at time t


def most_probable_run_length(R: np.ndarray) -> np.ndarray:
    """
    Most probable run length at each time step.

    Returns
    -------
    np.ndarray (T,) — argmax run length
    """
    return R[:, 1:].argmax(axis=0)


def detect_changepoints(
    data: np.ndarray,
    hazard: float = 1 / 50,
    threshold: float = 0.30,
    min_gap: int = 21,
) -> list:
    """
    Identify changepoint indices where P(changepoint) exceeds threshold.

    Parameters
    ----------
    data      : (T,) return series
    hazard    : per-step changepoint probability
    threshold : P(changepoint) threshold for flagging
    min_gap   : minimum days between consecutive changepoints

    Returns
    -------
    list of (index, probability) tuples
    """
    R = bocpd(data, hazard=hazard)
    cp_probs = changepoint_probability(R)

    changepoints = []
    last_cp = -min_gap

    for t, p in enumerate(cp_probs):
        if p >= threshold and t - last_cp >= min_gap:
            changepoints.append((t, round(float(p), 4)))
            last_cp = t

    return changepoints


# =============================================================================
# Validation Against Synthetic 2018/2020 Breakpoints
# =============================================================================

def validate_bocpd_on_crises(seed: int = 42) -> pd.DataFrame:
    """
    Validate BOCPD on synthetic data with known injected breakpoints.

    Simulates a 3-segment return series:
      Segment 1 (Risk-On):    Normal(0.0008, 0.008),  200 days
      Segment 2 (Risk-Off):   Normal(-0.0015, 0.035), 100 days  ← 2018 IL&FS shock
      Segment 3 (Post-Shock): Normal(-0.0005, 0.022), 150 days  ← 2020 COVID start
      Segment 4 (Recovery):   Normal(0.0010, 0.012),  100 days  ← recovery

    True changepoints at: t=200, t=300, t=450
    BOCPD should fire within ±10 days of each.

    Returns
    -------
    pd.DataFrame with: true_cp, detected_cp, lag, probability
    """
    rng = np.random.default_rng(seed)

    seg1 = rng.normal(0.0008, 0.008,  200)
    seg2 = rng.normal(-0.0015, 0.035, 100)
    seg3 = rng.normal(-0.0005, 0.022, 150)
    seg4 = rng.normal(0.0010, 0.012,  100)
    data = np.concatenate([seg1, seg2, seg3, seg4])

    true_breakpoints = [200, 300, 450]

    detected = detect_changepoints(data, hazard=1/50, threshold=0.20, min_gap=21)

    rows = []
    for true_cp in true_breakpoints:
        # Find closest detected changepoint
        if detected:
            dists = [(abs(d[0] - true_cp), d[0], d[1]) for d in detected]
            dists.sort()
            lag, det_cp, prob = dists[0]
        else:
            lag, det_cp, prob = None, None, 0.0

        rows.append({
            "true_cp":        true_cp,
            "detected_cp":    det_cp,
            "lag_days":       lag,
            "cp_probability": prob,
            "within_10_days": lag is not None and abs(lag) <= 10,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Streaming Dirichlet Posterior Update
# =============================================================================

class StreamingDirichletPosterior:
    """
    Online Dirichlet posterior for regime transition probability.

    Maintains a Dirichlet(alpha) posterior over K regime probabilities.
    Updates conjugately with each observed regime transition.

    Example
    -------
    >>> post = StreamingDirichletPosterior(K=5)
    >>> post.update(from_regime=0, to_regime=1)
    >>> print(post.posterior_mean())
    """

    def __init__(self, K: int = 5, alpha0: float = 1.0):
        self.K     = K
        self.alpha = np.full((K, K), alpha0, dtype=float)  # (K, K) Dirichlet params

    def update(self, from_regime: int, to_regime: int) -> "StreamingDirichletPosterior":
        """Update posterior with one observed transition."""
        self.alpha[from_regime, to_regime] += 1
        return self

    def update_from_sequence(self, states: np.ndarray) -> "StreamingDirichletPosterior":
        """Batch-update from a decoded state sequence."""
        for t in range(len(states) - 1):
            self.update(int(states[t]), int(states[t + 1]))
        return self

    def posterior_mean(self) -> np.ndarray:
        """Posterior mean transition matrix: E[P_{ij}] = alpha_{ij} / sum_j alpha_{ij}."""
        row_sums = self.alpha.sum(axis=1, keepdims=True)
        return self.alpha / (row_sums + 1e-12)

    def credible_interval(self, from_regime: int, to_regime: int, level: float = 0.95) -> tuple:
        """95% credible interval on P[from_regime, to_regime]."""
        a = self.alpha[from_regime, to_regime]
        b = self.alpha[from_regime].sum() - a
        lo = (1 - level) / 2
        hi = 1 - lo
        return (
            float(beta_dist.ppf(lo, a, b)),
            float(beta_dist.ppf(hi, a, b)),
        )

    def summary(self) -> pd.DataFrame:
        """Full posterior mean transition matrix as DataFrame."""
        return pd.DataFrame(
            self.posterior_mean(),
            index=[f"from_{k}" for k in range(self.K)],
            columns=[f"to_{k}" for k in range(self.K)],
        )


# =============================================================================
# Online/Batch Reconciliation Diagnostic
# =============================================================================

def online_batch_reconciliation(
    batch_probs: np.ndarray,
    online_probs: np.ndarray,
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """
    Compare online particle filter regime probabilities against batch HMM probs.

    Flags dates where |online_prob - batch_prob| > tolerance for any regime.

    Parameters
    ----------
    batch_probs  : (T, K) — batch-fitted HMM posterior probs
    online_probs : (T, K) — online particle filter regime probs
    tolerance    : deviation threshold

    Returns
    -------
    pd.DataFrame with per-date max deviation and flag
    """
    assert batch_probs.shape == online_probs.shape, "Shapes must match"
    deviation = np.abs(batch_probs - online_probs)
    max_dev   = deviation.max(axis=1)

    df = pd.DataFrame({
        "max_deviation":   max_dev,
        "flagged":         max_dev > tolerance,
        "dominant_batch":  batch_probs.argmax(axis=1),
        "dominant_online": online_probs.argmax(axis=1),
        "regime_mismatch": batch_probs.argmax(axis=1) != online_probs.argmax(axis=1),
    })

    return df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    rng = np.random.default_rng(42)

    print("=== BOCPD Validation on Synthetic Crisis Data ===")
    val_df = validate_bocpd_on_crises()
    print(val_df.to_string(index=False))

    print("\n=== Streaming Dirichlet Posterior ===")
    states = np.array([0, 0, 0, 1, 1, 2, 2, 1, 0, 0, 4, 4, 3, 2, 0, 0])
    post = StreamingDirichletPosterior(K=5)
    post.update_from_sequence(states)
    print("Posterior mean transition matrix:")
    print(post.summary().round(3).to_string())
    print(f"\nP[0→1] 95% CI: {post.credible_interval(0, 1)}")

    print("\n=== BOCPD on 200-day synthetic series ===")
    seg_a = rng.normal(0.001, 0.010, 100)
    seg_b = rng.normal(-0.005, 0.040, 100)
    data  = np.concatenate([seg_a, seg_b])
    R = bocpd(data, hazard=1/50)
    cp_probs = changepoint_probability(R)
    print(f"Max P(changepoint): {cp_probs.max():.4f} at t={cp_probs.argmax()}")
    print(f"True changepoint at t=100. Detection lag: {cp_probs.argmax() - 100} days")
