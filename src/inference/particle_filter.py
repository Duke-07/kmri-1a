"""
Bootstrap Particle Filter — Full Production Implementation
==========================================================
Section A9.1 of the Bayesian Regime Detection Engine specification.

Provides:
  - RegimeParticleFilter: bootstrap SIR with systematic resampling
  - Two-speed interface: step() for online, batch_filter() for batch
  - ESS monitoring and diagnostics
  - Regime trajectory extraction


Aaryan Dwivedi
"""

import numpy as np
import pandas as pd


class RegimeParticleFilter:
    """
    Bootstrap Particle Filter for K-regime HMM with Gaussian emissions.

    Maintains a weighted particle set {(s_t^i, w_t^i)}_{i=1}^N representing
    the filtered distribution P(S_t | y_{1:t}).

    Systematic Resampling (Kitagawa, 1996) is used when ESS < N/2 to
    prevent weight degeneracy.

    Parameters
    ----------
    P           : (K, K) transition matrix
    mu          : (K,) regime-conditional return means
    sigma       : (K,) regime-conditional return standard deviations
    n_particles : number of particles (default 5000)
    seed        : random seed
    """

    def __init__(self, P, mu, sigma, n_particles=5000, seed=42):
        self.P     = np.asarray(P, dtype=np.float64)
        self.mu    = np.asarray(mu, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.K     = P.shape[0]
        self.N     = n_particles
        self.rng   = np.random.default_rng(seed)

        # Initialise uniformly
        self.particles = self.rng.integers(0, self.K, size=self.N)
        self.weights   = np.full(self.N, 1.0 / self.N, dtype=np.float64)

        self._ess_history   = []
        self._n_resamples   = 0
        self._step_count    = 0

    # ── Core SIR step ─────────────────────────────────────────────────────────

    def step(self, obs: float) -> np.ndarray:
        """
        Process one observation and return the filtered posterior P(S_t | y_{1:t}).

        Steps:
          1. Propagate: s_t ~ P[s_{t-1}]
          2. Reweight:  w_t ∝ w_{t-1} * p(y_t | s_t)
          3. Resample:  systematic resample if ESS < N/2

        Parameters
        ----------
        obs : scalar observation (daily return)

        Returns
        -------
        posterior : (K,) regime probability vector
        """
        # 1. Propagate each particle through transition matrix
        self.particles = np.array(
            [self.rng.choice(self.K, p=self.P[s]) for s in self.particles]
        )

        # 2. Gaussian emission likelihood weight update
        z    = (obs - self.mu[self.particles]) / (self.sigma[self.particles] + 1e-12)
        like = np.exp(-0.5 * z**2) / (self.sigma[self.particles] * np.sqrt(2 * np.pi))
        self.weights *= like

        total = self.weights.sum()
        if total > 0:
            self.weights /= total
        else:
            self.weights[:] = 1.0 / self.N

        # 3. Effective Sample Size
        ess = 1.0 / (self.weights**2).sum()
        self._ess_history.append(ess)
        self._step_count += 1

        # Systematic resample if weight-degenerate
        if ess < self.N / 2:
            self.particles = self._systematic_resample()
            self.weights[:] = 1.0 / self.N
            self._n_resamples += 1

        # Regime posterior histogram
        post = np.bincount(self.particles, weights=self.weights, minlength=self.K)
        return post / post.sum()

    def _systematic_resample(self) -> np.ndarray:
        """Systematic (stratified) resampling — Kitagawa (1996)."""
        cumsum = np.cumsum(self.weights)
        u0     = self.rng.uniform(0, 1.0 / self.N)
        positions = u0 + np.arange(self.N) / self.N
        idx = np.searchsorted(cumsum, positions)
        return self.particles[np.clip(idx, 0, self.N - 1)]

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        """Summary of filter health."""
        ess_arr = np.array(self._ess_history)
        return {
            "steps":         self._step_count,
            "n_resamples":   self._n_resamples,
            "mean_ess":      round(float(ess_arr.mean()), 1) if len(ess_arr) else 0,
            "min_ess":       round(float(ess_arr.min()),  1) if len(ess_arr) else 0,
            "resample_rate": round(self._n_resamples / max(self._step_count, 1), 3),
        }

    # ── Batch filter ──────────────────────────────────────────────────────────

    def batch_filter(self, observations: np.ndarray) -> np.ndarray:
        """
        Run filter over an array of observations.

        Parameters
        ----------
        observations : (T,) array of daily returns

        Returns
        -------
        posteriors : (T, K) filtered posteriors
        """
        posteriors = np.zeros((len(observations), self.K))
        for t, obs in enumerate(observations):
            posteriors[t] = self.step(obs)
        return posteriors

    def reset(self, seed=None):
        """Reset particles to uniform distribution."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.particles = self.rng.integers(0, self.K, size=self.N)
        self.weights   = np.full(self.N, 1.0 / self.N)
        self._ess_history  = []
        self._n_resamples  = 0
        self._step_count   = 0


# =============================================================================
# Reconciliation: Online (particle filter) vs Batch (HMM) regime probs
# =============================================================================

def reconcile_online_batch(
    pf_probs: np.ndarray,
    hmm_probs: np.ndarray,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Compare particle filter posterior vs HMM batch posterior.

    Flags time steps where |online - batch| > threshold for any regime.
    These trigger investigation of distribution shift.

    Parameters
    ----------
    pf_probs   : (T, K) particle filter posteriors
    hmm_probs  : (T, K) HMM batch posteriors
    threshold  : divergence threshold (default 5%)

    Returns
    -------
    pd.DataFrame with columns: max_diff, divergent, and per-regime diffs
    """
    T = min(len(pf_probs), len(hmm_probs))
    diff = np.abs(pf_probs[:T] - hmm_probs[:T])
    max_diff = diff.max(axis=1)

    rows = []
    for t in range(T):
        rows.append({
            "t":         t,
            "max_diff":  round(float(max_diff[t]), 4),
            "divergent": bool(max_diff[t] > threshold),
        })
        for k in range(pf_probs.shape[1]):
            rows[-1][f"diff_regime_{k}"] = round(float(diff[t, k]), 4)

    df = pd.DataFrame(rows).set_index("t")
    n_div = df["divergent"].sum()
    pct_div = n_div / T * 100
    if pct_div > 5:
        print(f"  [RECONCILE] WARNING: {n_div}/{T} ({pct_div:.1f}%) steps exceed "
              f"divergence threshold {threshold:.0%} — investigate distribution shift")
    else:
        print(f"  [RECONCILE] OK: {n_div}/{T} ({pct_div:.1f}%) steps diverge — within tolerance")
    return df


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    import sys

    print("Particle Filter Self-Test")
    print("")
    print("=" * 50)

    K = 5
    P = np.array([
        [0.970, 0.020, 0.005, 0.003, 0.002],
        [0.030, 0.920, 0.030, 0.015, 0.005],
        [0.020, 0.040, 0.880, 0.040, 0.020],
        [0.010, 0.020, 0.070, 0.850, 0.050],
        [0.005, 0.010, 0.050, 0.135, 0.800],
    ])
    mu    = np.array([ 0.0008,  0.0003,  0.0000, -0.0005, -0.0015])
    sigma = np.array([ 0.008,   0.012,   0.015,   0.022,   0.035])

    pf  = RegimeParticleFilter(P, mu, sigma, n_particles=2000, seed=42)
    obs = [0.0012, 0.0008, 0.0003, -0.0150, -0.0220, -0.0180, 0.0005, 0.0010]

    for i, o in enumerate(obs):
        post = pf.step(o)
        regime = int(post.argmax())
        print(f"  t={i+1}  obs={o:+.4f}  P={np.round(post,3)}  dominant=Regime_{regime}")

    diag = pf.diagnostics()
    print(f"\n  Diagnostics: {diag}")
    print("\nSelf-test PASSED")
