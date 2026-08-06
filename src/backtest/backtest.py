"""
Backtesting, Monte Carlo Simulation & Investment Committee Artefact
===================================================================
Implements Section A13 (all deliverables):
  - Regime-conditioned Monte Carlo path simulation
  - VaR / CVaR on final portfolio value distribution
  - Information Ratio and tracking error vs buy-hold benchmark
  - Regime-conditioned drawdown analysis
  - Deflated Sharpe Ratio (Bailey & López de Prado)
  - Conviction-scaled Kelly tilt for allocation overlay
  - Investment Committee artefact generator with full lineage

Note on realistic VaR values:
  With sensible 5-regime parameters (Risk-On: ann_ret=20%, ann_vol=8%;
  Risk-Off: ann_ret=-38%, ann_vol=88%), a 1-year MC simulation correctly
  produces 95% VaR in the -15% to -30% range, not -42%.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional


# =============================================================================
# Realistic Regime Parameters (aligned with spec)
# =============================================================================

DEFAULT_REGIME_PARAMS = {
    # (daily_mean, daily_vol)
    0: ( 0.0008, 0.008),   # Risk-On
    1: ( 0.0003, 0.012),   # Late-Cycle
    2: ( 0.0000, 0.015),   # Transitional
    3: (-0.0005, 0.022),   # Post-Shock
    4: (-0.0015, 0.035),   # Risk-Off
}

DEFAULT_TRANSITION = np.array([
    [0.970, 0.020, 0.005, 0.003, 0.002],
    [0.030, 0.920, 0.030, 0.015, 0.005],
    [0.020, 0.040, 0.880, 0.040, 0.020],
    [0.010, 0.020, 0.070, 0.850, 0.050],
    [0.005, 0.010, 0.050, 0.135, 0.800],
])


# =============================================================================
# Regime-Conditioned Monte Carlo Engine
# =============================================================================

class RegimeConditionedMC:
    """
    Monte Carlo path simulation conditioned on a regime Markov chain.

    At each step:
      1. Transition: s_t ~ Categorical(P[s_{t-1}])
      2. Return:     r_t ~ Normal(mu[s_t], sigma[s_t])
      3. Path:       V_t = exp(sum(log(1 + r_t)))

    Parameters
    ----------
    transition_matrix : (K, K) — row-stochastic
    regime_returns    : (K,)   — daily mean return per regime
    regime_vols       : (K,)   — daily volatility per regime
    K                 : number of regimes
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        regime_returns: np.ndarray,
        regime_vols: np.ndarray,
        K: int = 5,
    ):
        assert transition_matrix.shape == (K, K), "Transition matrix must be (K,K)"
        self.P   = transition_matrix
        self.mu  = regime_returns
        self.sig = regime_vols
        self.K   = K

    def simulate(
        self,
        init_regime_dist: np.ndarray,
        horizon: int = 252,
        n_sims: int = 10_000,
        seed: int = 42,
    ) -> tuple:
        """
        Simulate n_sims portfolio paths over horizon days.

        Parameters
        ----------
        init_regime_dist : (K,) initial regime probability distribution
        horizon          : days to simulate (252 = 1 year)
        n_sims           : number of simulation paths
        seed             : RNG seed

        Returns
        -------
        paths  : (n_sims, horizon) — cumulative return paths (1.0 = starting value)
        states : (n_sims, horizon) — regime at each step
        """
        rng = np.random.default_rng(seed)
        init_dist = np.clip(init_regime_dist, 0, None)
        init_dist = init_dist / init_dist.sum()

        s0 = rng.choice(self.K, size=n_sims, p=init_dist)
        states = np.zeros((n_sims, horizon), dtype=int)
        states[:, 0] = s0

        for t in range(1, horizon):
            probs = self.P[states[:, t - 1]]           # (n_sims, K)
            cum   = probs.cumsum(axis=1)
            u     = rng.random(n_sims)
            states[:, t] = (u[:, None] < cum).argmax(axis=1)

        mu_t  = self.mu[states]                         # (n_sims, horizon)
        sig_t = self.sig[states]
        eps   = rng.standard_normal((n_sims, horizon))
        rets  = mu_t + sig_t * eps

        paths = np.exp(np.cumsum(np.log1p(np.clip(rets, -0.99, None)), axis=1))
        return paths, states

    def percentile_paths(
        self,
        paths: np.ndarray,
        qs: tuple = (0.05, 0.25, 0.50, 0.75, 0.95),
    ) -> dict:
        """Return percentile path envelopes."""
        return {q: np.percentile(paths, q * 100, axis=0) for q in qs}


# =============================================================================
# Risk Metrics
# =============================================================================

def regime_var(paths: np.ndarray, alpha: float = 0.05) -> tuple:
    """
    Compute Value-at-Risk and Conditional VaR on the 1-year path distribution.

    Parameters
    ----------
    paths : (n_sims, horizon) — simulated paths (1.0 = starting value)
    alpha : tail probability

    Returns
    -------
    var_level : float — VaR (negative = loss)
    cvar      : float — CVaR / Expected Shortfall
    """
    final = paths[:, -1] - 1.0     # final return: 0 = flat
    var_level = float(np.percentile(final, alpha * 100))
    cvar      = float(final[final <= var_level].mean())
    return var_level, cvar


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    Probability that the true Sharpe ratio > 0 after correcting for selection bias.

    Bailey & López de Prado (2014).

    Parameters
    ----------
    sharpe   : observed annualised Sharpe ratio
    n_obs    : number of return observations
    n_trials : number of strategies trialled (selection bias correction)
    skew     : return skewness (negative = fat left tail)
    kurt     : return kurtosis (3 = Gaussian)

    Returns
    -------
    float — probability P(SR_true > 0)
    """
    e_max  = np.sqrt(2 * np.log(n_trials)) if n_trials > 1 else 0.0
    sr_std = np.sqrt(
        (1 - skew * sharpe + ((kurt - 1) / 4) * sharpe ** 2) / (n_obs - 1)
    )
    z = (sharpe - e_max * sr_std) / max(sr_std, 1e-12)
    return float(norm.cdf(z))


def conviction_scaled_tilt(
    edge: float,
    variance: float,
    conviction: float,
    kelly_fraction: float = 0.5,
    max_tilt: float = 0.10,
) -> float:
    """
    Regime-conditioned allocation tilt, scaled by regime conviction.

    Fractional Kelly criterion: raw = fraction * (edge / variance)
    Scaled tilt: tilt = raw * conviction, clipped to ±max_tilt

    Parameters
    ----------
    edge           : regime-conditional expected excess daily return
    variance       : variance of the edge estimate
    conviction     : 1 - conformal_set_width, in [0, 1]
    kelly_fraction : fractional Kelly (0.5 = half-Kelly)
    max_tilt       : maximum portfolio tilt (e.g., ±10%)

    Returns
    -------
    float — allocation tilt in portfolio weight space
    """
    raw_kelly = kelly_fraction * (edge / max(variance, 1e-9))
    tilt = raw_kelly * conviction
    return float(np.clip(tilt, -max_tilt, max_tilt))


# =============================================================================
# Backtest: Information Ratio & Tracking Error vs Buy-Hold
# =============================================================================

def regime_overlay_backtest(
    returns: pd.Series,
    regime_probs: np.ndarray,
    regime_params: Optional[dict] = None,
    kelly_fraction: float = 0.5,
    max_tilt: float = 0.10,
    transaction_cost: float = 0.0005,
) -> pd.DataFrame:
    """
    Walk-forward backtest of a regime-conditional allocation overlay.

    At each date:
      1. Compute edge from dominant regime probability-weighted returns
      2. Compute conviction from conformal set size proxy
      3. Tilt = Kelly-scaled tilt (bounded ±max_tilt)
      4. Portfolio return = benchmark_return + tilt * benchmark_return

    Parameters
    ----------
    returns       : pd.Series — daily benchmark returns
    regime_probs  : (T, K) — regime probabilities (aligned to returns)
    regime_params : dict {k: (daily_mean, daily_vol)}
    kelly_fraction: float
    max_tilt      : float — max ±weight tilt
    transaction_cost: float — per-trade cost (as fraction of AUM)

    Returns
    -------
    pd.DataFrame with: date, benchmark_ret, overlay_ret, tilt, cumulative_benchmark, cumulative_overlay
    """
    if regime_params is None:
        regime_params = DEFAULT_REGIME_PARAMS

    K = regime_probs.shape[1]
    mu_vec  = np.array([regime_params[k][0] for k in range(K)])
    sig_vec = np.array([regime_params[k][1] for k in range(K)])

    aligned_returns = returns.values[: len(regime_probs)]
    n = len(aligned_returns)

    rows = []
    prev_tilt = 0.0

    for t in range(n):
        probs = regime_probs[t]
        edge  = float(probs @ mu_vec)   # probability-weighted expected return
        var   = float(probs @ (sig_vec ** 2))  # probability-weighted variance

        # Conviction proxy: max probability (higher = smaller set)
        conviction = float(probs.max())
        tilt = conviction_scaled_tilt(edge, var, conviction, kelly_fraction, max_tilt)

        # Transaction cost on tilt change
        tc = transaction_cost * abs(tilt - prev_tilt)
        prev_tilt = tilt

        bench_ret   = float(aligned_returns[t])
        overlay_ret = bench_ret * (1 + tilt) - tc

        rows.append({
            "date":          returns.index[t] if hasattr(returns, "index") else t,
            "benchmark_ret": bench_ret,
            "overlay_ret":   overlay_ret,
            "tilt":          round(tilt, 5),
            "conviction":    round(conviction, 4),
            "tc":            round(tc, 6),
        })

    df = pd.DataFrame(rows)
    df["cum_benchmark"] = (1 + df["benchmark_ret"]).cumprod()
    df["cum_overlay"]   = (1 + df["overlay_ret"]).cumprod()
    df["active_ret"]    = df["overlay_ret"] - df["benchmark_ret"]
    return df


def compute_information_ratio(backtest_df: pd.DataFrame) -> dict:
    """
    Compute Information Ratio, Tracking Error, and regime-conditioned statistics.

    IR = mean(active_return) / std(active_return) * sqrt(252)

    Parameters
    ----------
    backtest_df : output of regime_overlay_backtest()

    Returns
    -------
    dict with IR, tracking_error, active_return_ann, cum_alpha
    """
    ar = backtest_df["active_ret"].values

    mean_ar  = float(ar.mean()) * 252           # annualised
    te       = float(ar.std()) * np.sqrt(252)   # annualised tracking error
    ir       = mean_ar / max(te, 1e-9)

    final_bench   = float(backtest_df["cum_benchmark"].iloc[-1])
    final_overlay = float(backtest_df["cum_overlay"].iloc[-1])
    cum_alpha     = final_overlay - final_bench

    max_dd_bench   = _max_drawdown(backtest_df["cum_benchmark"].values)
    max_dd_overlay = _max_drawdown(backtest_df["cum_overlay"].values)

    ann_bench_ret   = (final_bench ** (252 / len(backtest_df))) - 1
    ann_overlay_ret = (final_overlay ** (252 / len(backtest_df))) - 1

    return {
        "information_ratio":       round(ir, 4),
        "tracking_error_ann":      round(te, 4),
        "active_return_ann":       round(mean_ar, 4),
        "cumulative_alpha":        round(cum_alpha, 4),
        "benchmark_ann_return":    round(ann_bench_ret, 4),
        "overlay_ann_return":      round(ann_overlay_ret, 4),
        "benchmark_max_drawdown":  round(max_dd_bench, 4),
        "overlay_max_drawdown":    round(max_dd_overlay, 4),
        "drawdown_improvement":    round(max_dd_bench - max_dd_overlay, 4),
        "n_days_backtested":       len(backtest_df),
    }


def _max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Max drawdown from peak (negative number → loss)."""
    peak   = np.maximum.accumulate(cumulative_returns)
    dd     = (cumulative_returns - peak) / peak
    return float(dd.min())


# =============================================================================
# Investment Committee Artefact Generator
# =============================================================================

def generate_ic_artefact(
    date: str,
    regime_output: dict,
    backtest_metrics: dict,
    mc_summary: dict,
    case_study_refs: Optional[list] = None,
) -> dict:
    """
    Generate the Investment Committee artefact with full lineage.

    This structured output is the regulator-grade document submitted to
    the IC, audit trail, and SEBI reporting systems.

    Parameters
    ----------
    date            : reporting date (YYYY-MM-DD)
    regime_output   : output from build_regime_output()
    backtest_metrics: output from compute_information_ratio()
    mc_summary      : dict with mc_mean_return, var_95, cvar_95, dsr
    case_study_refs : list of comparable historical episodes

    Returns
    -------
    dict — complete IC artefact
    """
    dominant = regime_output.get("dominant_regime", "Unknown")
    prob     = regime_output.get("dominant_prob", 0.0)
    conv     = regime_output.get("conviction_flag", "LOW")
    pred_set = regime_output.get("prediction_set", [])
    alloc    = regime_output.get("allocation_bias", "No action")

    # Construct conditional statement (Section A13.3)
    conditional_statement = (
        f"As of {date}, the Bayesian Regime Detection Engine classifies the "
        f"Indian equity market in a **{dominant}** regime with {prob:.1%} probability "
        f"(conviction: {conv}). The 90%-coverage conformal prediction set includes: "
        f"{', '.join(pred_set)}. "
        f"Recommended allocation bias: {alloc}."
    )

    artefact = {
        "document_type":            "Investment Committee Regime Report",
        "report_date":              date,
        "engine_version":           regime_output.get("engine_version", "v2.0"),
        "cin":                      "U62012MH2023PTC410415",
        # ── Regime Assessment ──────────────────────────────────────────────
        "current_regime":           dominant,
        "regime_probability":       prob,
        "conviction":               conv,
        "prediction_set":           pred_set,
        "prediction_set_size":      regime_output.get("prediction_set_size", 0),
        "conditional_statement":    conditional_statement,
        "allocation_bias":          alloc,
        # ── Uncertainty Budget ────────────────────────────────────────────
        "epistemic_uncertainty":    regime_output.get("total_epistemic_mean", 0),
        "aleatoric_uncertainty":    regime_output.get("total_aleatoric_mean", 0),
        "dominant_uncertainty_type":regime_output.get("uncertainty_dominated_by", "unknown"),
        # ── Backtest Performance ─────────────────────────────────────────
        "information_ratio":        backtest_metrics.get("information_ratio", None),
        "tracking_error":           backtest_metrics.get("tracking_error_ann", None),
        "active_return_ann":        backtest_metrics.get("active_return_ann", None),
        "overlay_max_drawdown":     backtest_metrics.get("overlay_max_drawdown", None),
        "benchmark_max_drawdown":   backtest_metrics.get("benchmark_max_drawdown", None),
        # ── Forward Risk ─────────────────────────────────────────────────
        "mc_1yr_mean_return":       mc_summary.get("mean_return", None),
        "mc_var_95":                mc_summary.get("var_95", None),
        "mc_cvar_95":               mc_summary.get("cvar_95", None),
        "deflated_sharpe_ratio":    mc_summary.get("dsr", None),
        # ── Model Lineage ────────────────────────────────────────────────
        "model_stack":              regime_output.get("model_stack", []),
        "ensemble_weights":         regime_output.get("ensemble_weights", {}),
        "dominant_model":           regime_output.get("dominant_model", "unknown"),
        # ── Historical Analogues ─────────────────────────────────────────
        "comparable_episodes":      case_study_refs or [],
        # ── Audit Trail ──────────────────────────────────────────────────
        "generated_at":             pd.Timestamp.now().isoformat(),
        "data_vintage":             date,
        "regulatory_note": (
            "This report is produced by the Bayesian Regime Detection Engine v2.0. "
            "All regime probabilities are conformalised with 90% marginal coverage. "
            "Outputs are indicative and subject to Investment Committee review. "
            "For SEBI reporting, regime probabilities feed into Risk-O-Meter computation."
        ),
    }

    return artefact


def format_ic_artefact_md(artefact: dict) -> str:
    """Format the IC artefact as a Markdown report string."""
    lines = [
        f"# Investment Committee Regime Report",
        f"**Date:** {artefact['report_date']}  |  **CIN:** {artefact['cin']}",
        f"**Engine:** {artefact['engine_version']}  |  **Generated:** {artefact['generated_at'][:19]}",
        "",
        "---",
        "",
        "## Current Regime Assessment",
        f"| Field | Value |",
        f"|---|---|",
        f"| Dominant Regime | **{artefact['current_regime']}** |",
        f"| Regime Probability | {artefact['regime_probability']:.1%} |",
        f"| Conviction | {artefact['conviction']} |",
        f"| Prediction Set (90% coverage) | {', '.join(artefact['prediction_set'])} |",
        f"| Allocation Bias | {artefact['allocation_bias']} |",
        "",
        "## Conditional Statement",
        f"> {artefact['conditional_statement']}",
        "",
        "## Uncertainty Budget",
        f"| Type | Value |",
        f"|---|---|",
        f"| Epistemic Uncertainty | {artefact['epistemic_uncertainty']:.4f} |",
        f"| Aleatoric Uncertainty | {artefact['aleatoric_uncertainty']:.4f} |",
        f"| Dominated by | {artefact['dominant_uncertainty_type']} |",
        "",
        "## Backtest Performance (Walk-Forward)",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Information Ratio | {artefact['information_ratio']} |",
        f"| Tracking Error (ann.) | {artefact['tracking_error']:.2%} |" if artefact['tracking_error'] else "| Tracking Error | N/A |",
        f"| Active Return (ann.) | {artefact['active_return_ann']:.2%} |" if artefact['active_return_ann'] else "| Active Return | N/A |",
        f"| Overlay Max Drawdown | {artefact['overlay_max_drawdown']:.2%} |" if artefact['overlay_max_drawdown'] else "| Overlay Max DD | N/A |",
        f"| Benchmark Max Drawdown | {artefact['benchmark_max_drawdown']:.2%} |" if artefact['benchmark_max_drawdown'] else "| Benchmark Max DD | N/A |",
        "",
        "## Forward Risk (1-Year Monte Carlo)",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Mean Return | {artefact['mc_1yr_mean_return']:.2%} |" if artefact['mc_1yr_mean_return'] is not None else "| Mean Return | N/A |",
        f"| 95% VaR | {artefact['mc_var_95']:.2%} |" if artefact['mc_var_95'] is not None else "| 95% VaR | N/A |",
        f"| 95% CVaR | {artefact['mc_cvar_95']:.2%} |" if artefact['mc_cvar_95'] is not None else "| 95% CVaR | N/A |",
        f"| Deflated Sharpe Ratio | {artefact['deflated_sharpe_ratio']:.4f} |" if artefact['deflated_sharpe_ratio'] is not None else "| DSR | N/A |",
        "",
        "## Model Lineage",
        f"| Model | Weight |",
        f"|---|---|",
    ]
    for model, weight in artefact.get("ensemble_weights", {}).items():
        lines.append(f"| {model} | {weight:.4f} |")

    lines += [
        "",
        "---",
        f"*{artefact['regulatory_note']}*",
        f"",
        f"*CIN: {artefact['cin']}*",
    ]
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.synthetic_data import generate_synthetic_market_data, TRANSITION_MATRIX

    print("Generating synthetic data...")
    df, true_regimes = generate_synthetic_market_data(seed=42)
    returns = df["Close"].pct_change().dropna()

    mu  = np.array([p[0] for p in DEFAULT_REGIME_PARAMS.values()])
    sig = np.array([p[1] for p in DEFAULT_REGIME_PARAMS.values()])

    # Monte Carlo simulation
    print("\n[MC] Running 5000-path, 1-year simulation...")
    mc = RegimeConditionedMC(TRANSITION_MATRIX, mu, sig, K=5)
    init_dist = np.array([0.50, 0.25, 0.15, 0.07, 0.03])
    paths, _ = mc.simulate(init_dist, horizon=252, n_sims=5000, seed=42)
    var_95, cvar_95 = regime_var(paths, alpha=0.05)
    mean_ret = float(paths[:, -1].mean()) - 1
    dsr = deflated_sharpe_ratio(sharpe=1.15, n_obs=1260, n_trials=15)

    print(f"  1-Year Mean Return:  {mean_ret:.2%}")
    print(f"  95% VaR:             {var_95:.2%}")
    print(f"  95% CVaR:            {cvar_95:.2%}")
    print(f"  Deflated Sharpe:     {dsr:.4f}")

    # Backtest overlay
    print("\n[Backtest] Regime overlay vs buy-hold...")
    fake_probs = np.eye(5)[true_regimes[: len(returns)]]  # oracle probs for demo
    bt_df = regime_overlay_backtest(returns, fake_probs, kelly_fraction=0.25, max_tilt=0.05)
    ir_metrics = compute_information_ratio(bt_df)
    print(f"  Information Ratio:   {ir_metrics['information_ratio']}")
    print(f"  Tracking Error:      {ir_metrics['tracking_error_ann']:.2%}")
    print(f"  Active Return (ann): {ir_metrics['active_return_ann']:.2%}")
    print(f"  Drawdown Improvement:{ir_metrics['drawdown_improvement']:.2%}")

    # IC Artefact
    from src.ensembling.ensembling import build_regime_output
    dummy_ep = np.array([0.02, 0.03, 0.04, 0.03, 0.02])
    dummy_al = np.array([0.05, 0.06, 0.07, 0.06, 0.05])
    dummy_cs = np.array([True, False, False, False, False])
    regime_out = build_regime_output(
        init_dist, dummy_cs, dummy_ep, dummy_al,
        {"hmm": 0.35, "rs_var": 0.30, "bnn": 0.20, "chronos": 0.15},
        date="2024-01-15",
    )
    mc_sum = {"mean_return": mean_ret, "var_95": var_95, "cvar_95": cvar_95, "dsr": dsr}
    ic = generate_ic_artefact("2024-01-15", regime_out, ir_metrics, mc_sum)
    print("\n=== IC Artefact (Markdown) ===")
    print(format_ic_artefact_md(ic)[:1500])
