"""
Synthetic Indian Equity Market Data Generator
=============================================
Simulates 18 years (2007-2024) of daily market data across 5 economic regimes:
  0: Risk-On       | 1: Late-Cycle  | 2: Transitional
  3: Post-Shock    | 4: Risk-Off

Returns are Student-t distributed (fat tails) per regime.
Includes: Nifty 50, Midcap 100, Smallcap 100, India VIX, USD/INR, 10Y Gilt,
          AAA spread, FII/DII flows, monthly SIP, Advances/Declines,
          New Highs/Lows, % Above 50-DMA.
"""

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

# ---------------------------------------------------------------------------
# 5-Regime Specification (Hamilton 1989 style)
# ---------------------------------------------------------------------------

REGIME_NAMES = {
    0: "Risk-On",
    1: "Late-Cycle",
    2: "Transitional",
    3: "Post-Shock",
    4: "Risk-Off",
}

# Daily return: (mean, std, degrees_of_freedom)
# Risk-Off has fat tails (df=4); Risk-On close to Gaussian (df=20)
REGIME_RETURN_PARAMS = {
    0: ( 0.0008, 0.008, 20),   # Risk-On: positive drift, low vol
    1: ( 0.0003, 0.012, 12),   # Late-Cycle: slowing, moderate vol
    2: ( 0.0000, 0.015,  8),   # Transitional: no drift, elevated vol
    3: (-0.0005, 0.022,  6),   # Post-Shock: negative drift, high vol
    4: (-0.0015, 0.035,  4),   # Risk-Off: severe drawdown, fat tails
}

# Asymmetric transition matrix: regimes are sticky; Risk-Off -> Post-Shock likely
# Rows sum to 1; P[i, j] = P(next=j | current=i)
TRANSITION_MATRIX = np.array([
    [0.970, 0.020, 0.005, 0.003, 0.002],  # from Risk-On
    [0.030, 0.920, 0.030, 0.015, 0.005],  # from Late-Cycle
    [0.020, 0.040, 0.880, 0.040, 0.020],  # from Transitional
    [0.010, 0.020, 0.070, 0.850, 0.050],  # from Post-Shock
    [0.005, 0.010, 0.050, 0.135, 0.800],  # from Risk-Off
])

# India VIX per regime: (mean, std)
VIX_PARAMS = {
    0: (12.0, 1.5),
    1: (16.0, 2.0),
    2: (20.0, 2.5),
    3: (28.0, 4.0),
    4: (38.0, 8.0),
}

# Midcap/Smallcap return amplification + idiosyncratic vol
MC_AMP, MC_IDIO  = 1.25, 0.008
SC_AMP, SC_IDIO  = 1.55, 0.013

# FII/DII flows per regime (crore INR daily): (mean, std)
FII_PARAMS = {
    0: ( 600, 900),
    1: ( 150, 800),
    2: (-100, 1000),
    3: (-1500, 1200),
    4: (-2500, 1500),
}
DII_PARAMS = {
    0: ( 200,  500),
    1: ( 400,  600),
    2: ( 800,  700),
    3: (1600,  900),
    4: (2200, 1000),
}

# USD/INR drift per regime
INR_DAILY_DRIFT = {0: 0.00005, 1: 0.00008, 2: 0.00012, 3: 0.00025, 4: 0.00045}

# 10Y Gilt drift per regime (daily bps)
GILT_DAILY_DRIFT = {0: 0.0, 1: 0.002, 2: 0.004, 3: -0.010, 4: -0.015}

# Breadth: advances fraction
BREADTH_PARAMS = {
    0: (0.72, 0.08),
    1: (0.58, 0.09),
    2: (0.50, 0.10),
    3: (0.38, 0.10),
    4: (0.25, 0.09),
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_synthetic_market_data(
    start_date: str = "2007-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic daily Indian equity market data with 5 regimes.

    Returns
    -------
    df : pd.DataFrame  — multi-column market data indexed by business date
    regimes : np.ndarray (T,)  — true latent regime at each date
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    T = len(dates)

    # ------------------------------------------------------------------
    # 1. Simulate hidden Markov chain
    # ------------------------------------------------------------------
    regimes = np.empty(T, dtype=int)
    regimes[0] = 0  # start in Risk-On
    for t in range(1, T):
        regimes[t] = rng.choice(5, p=TRANSITION_MATRIX[regimes[t - 1]])

    # ------------------------------------------------------------------
    # 2. Regime-conditional returns (Student-t)
    # ------------------------------------------------------------------
    nifty_rets = np.zeros(T)
    for t in range(T):
        mu, sig, df = REGIME_RETURN_PARAMS[regimes[t]]
        nifty_rets[t] = mu + sig * rng.standard_t(df)

    # clip to avoid impossible daily moves (±20%)
    nifty_rets = np.clip(nifty_rets, -0.20, 0.20)

    nifty_close = 1000.0 * np.exp(np.cumsum(nifty_rets))

    # Midcap and Smallcap
    mc_rets = nifty_rets * MC_AMP + rng.normal(0, MC_IDIO, T)
    sc_rets = nifty_rets * SC_AMP + rng.normal(0, SC_IDIO, T)
    midcap_close = 1000.0 * np.exp(np.cumsum(mc_rets))
    smallcap_close = 1000.0 * np.exp(np.cumsum(sc_rets))

    # ------------------------------------------------------------------
    # 3. India VIX
    # ------------------------------------------------------------------
    vix = np.array([
        max(8, rng.normal(*VIX_PARAMS[regimes[t]])) for t in range(T)
    ])
    vix = np.clip(vix, 8, 90)

    # ------------------------------------------------------------------
    # 4. Breadth (Advances / Declines / % above 50-DMA)
    # ------------------------------------------------------------------
    adv_frac = np.array([rng.normal(*BREADTH_PARAMS[regimes[t]]) for t in range(T)])
    adv_frac = np.clip(adv_frac, 0.05, 0.95)
    total_stocks = 2000
    advances = (adv_frac * total_stocks).astype(int)
    declines = total_stocks - advances

    new_highs = (advances * rng.uniform(0.03, 0.08, T)).astype(int)
    new_lows = (declines * rng.uniform(0.03, 0.08, T)).astype(int)

    pct_above_50dma = np.clip(adv_frac * 100 + rng.normal(0, 3, T), 0, 100)

    # ------------------------------------------------------------------
    # 5. FII/DII flows
    # ------------------------------------------------------------------
    fii = np.array([rng.normal(*FII_PARAMS[regimes[t]]) for t in range(T)])
    dii = np.array([rng.normal(*DII_PARAMS[regimes[t]]) for t in range(T)])

    # ------------------------------------------------------------------
    # 6. Macro: USD/INR, 10Y Gilt, AAA spread
    # ------------------------------------------------------------------
    inr_daily = np.array([rng.normal(INR_DAILY_DRIFT[regimes[t]], 0.003) for t in range(T)])
    usd_inr = 45.0 * np.exp(np.cumsum(inr_daily))

    gilt_shock = np.array([rng.normal(GILT_DAILY_DRIFT[regimes[t]], 0.04) for t in range(T)])
    gilt_10y = 7.0 + np.cumsum(gilt_shock)
    gilt_10y = np.clip(gilt_10y, 4.0, 12.0)

    credit_spread = np.array([rng.normal(0.75 + 0.30 * (regimes[t] >= 3), 0.10) for t in range(T)])
    aaa_10y = gilt_10y + credit_spread

    # ------------------------------------------------------------------
    # 7. Monthly SIP (growing structurally)
    # ------------------------------------------------------------------
    sip_monthly = np.linspace(3_000, 26_000, T) + rng.normal(0, 400, T)

    # ------------------------------------------------------------------
    # 8. Assemble DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        "Date": dates,
        "Close": nifty_close,
        "Midcap_Close": midcap_close,
        "Smallcap_Close": smallcap_close,
        "IndiaVIX": vix,
        "Advances": advances,
        "Declines": declines,
        "NewHighs": new_highs,
        "NewLows": new_lows,
        "PctAbove50DMA": pct_above_50dma,
        "FII_Equity": fii,
        "DII_Equity": dii,
        "USDINR": usd_inr,
        "Gilt10Y": gilt_10y,
        "AAA10Y": aaa_10y,
        "SIP_Monthly": sip_monthly,
        "TrueRegime": regimes,
    }).set_index("Date")

    return df, regimes


# ---------------------------------------------------------------------------
# Regime summary helper
# ---------------------------------------------------------------------------

def regime_summary(regimes: np.ndarray) -> pd.DataFrame:
    """Return counts, pct, and mean duration per regime."""
    rows = []
    for k in range(5):
        mask = regimes == k
        count = mask.sum()
        runs = []
        cur = 0
        for v in mask:
            if v:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur)
                    cur = 0
        if cur > 0:
            runs.append(cur)
        rows.append({
            "Regime": REGIME_NAMES[k],
            "Days": count,
            "Pct": f"{100 * count / len(regimes):.1f}%",
            "MeanDuration": f"{np.mean(runs):.1f}" if runs else "N/A",
            "MaxDuration": max(runs) if runs else 0,
        })
    return pd.DataFrame(rows).set_index("Regime")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df, regimes = generate_synthetic_market_data(seed=42)
    out_path = __file__.replace("synthetic_data.py", "synthetic_indian_market.csv")
    df.drop(columns=["TrueRegime"]).to_csv(out_path)
    print(f"Saved {len(df)} rows to {out_path}")
    print("\nRegime Summary:")
    print(regime_summary(regimes).to_string())
    print(f"\nTransition matrix row sums: {TRANSITION_MATRIX.sum(axis=1)}")
