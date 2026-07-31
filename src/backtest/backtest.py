import numpy as np

class RegimeConditionedMC:
    def __init__(self, transition_matrix, regime_returns, regime_vols, K=5):
        self.P = transition_matrix # (k, k)
        self.mu = regime_returns   # (k,) daily mean returns per regime
        self.sig = regime_vols     # (k,) daily vol per regime
        self.K = K
        
    def simulate(self, init_regime_dist, horizon=252, n_sims=10000, seed=42):
        rng = np.random.default_rng(seed)
        s0 = rng.choice(self.K, size=n_sims, p=init_regime_dist)
        states = np.zeros((n_sims, horizon), dtype=int)
        states[:, 0] = s0
        
        for t in range(1, horizon):
            probs = self.P[states[:, t-1]]
            cum = probs.cumsum(axis=1)
            u = rng.random(n_sims)
            states[:, t] = (u[:, None] < cum).argmax(axis=1)
            
        mu_t = self.mu[states]
        sig_t = self.sig[states]
        eps = rng.standard_normal((n_sims, horizon))
        rets = mu_t + sig_t * eps
        
        paths = np.exp(np.cumsum(np.log1p(rets), axis=1))
        return paths, states
        
    def percentile_paths(self, paths, qs=(0.05, 0.25, 0.50, 0.75, 0.95)):
        return {q: np.percentile(paths, q*100, axis=0) for q in qs}

def regime_var(paths, alpha=0.05):
    """value-at-risk on final portfolio value distribution."""
    final = paths[:, -1] - 1.0
    var = np.percentile(final, alpha * 100)
    cvar = final[final <= var].mean()
    return var, cvar

def conviction_scaled_tilt(edge, variance, conviction, kelly_fraction=0.5, max_tilt=0.10):
    """
    edge: regime-conditional expected excess return; variance: its variance;
    conviction: 1 - conformal_set_width in [0,1].
    """
    raw = kelly_fraction * (edge / max(variance, 1e-9))
    tilt = raw * conviction
    return float(np.clip(tilt, -max_tilt, max_tilt))

def deflated_sharpe_ratio(sharpe, n_obs, n_trials, skew=0.0, kurt=3.0):
    """probability the true Sharpe > 0 after correcting for selection."""
    from scipy.stats import norm
    e_max = (np.sqrt(2 * np.log(n_trials)) if n_trials > 1 else 0.0)
    sr_std = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2) / (n_obs - 1))
    z = (sharpe - e_max * sr_std) / max(sr_std, 1e-12)
    return float(norm.cdf(z))

if __name__ == "__main__":
    P = np.array([
        [0.95, 0.03, 0.02, 0.00, 0.00],
        [0.05, 0.90, 0.05, 0.00, 0.00],
        [0.02, 0.08, 0.85, 0.03, 0.02],
        [0.00, 0.00, 0.10, 0.80, 0.10],
        [0.00, 0.00, 0.05, 0.05, 0.90]
    ])
    means = np.array([0.0008, 0.0002, 0.0000, -0.0010, -0.0025])
    vols = np.array([0.008, 0.012, 0.015, 0.022, 0.035])
    
    mc = RegimeConditionedMC(P, means, vols, K=5)
    init_dist = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    paths, states = mc.simulate(init_dist, horizon=252, n_sims=1000)
    var_95, cvar_95 = regime_var(paths, alpha=0.05)
    print(f"1-Year Monte Carlo Simulation (1000 paths):")
    print(f"  Final Mean Return: {paths[:, -1].mean() - 1:.4%}")
    print(f"  Regime-Conditioned VaR (95%): {var_95:.4%}")
    print(f"  Regime-Conditioned CVaR (95%): {cvar_95:.4%}")
    
    dsr = deflated_sharpe_ratio(sharpe=1.2, n_obs=1260, n_trials=20)
    print(f"Deflated Sharpe Ratio (Sharpe 1.2 over 5yrs, 20 trials): {dsr:.4f}")

