"""
Feature Engineering for Bayesian Regime Detection
==================================================
Implements the full feature pipeline from the project specification:
  Section A4.5  — Core regime features (returns, trend, vol, breadth, macro, flows)
  Section A7.2  — Topological Data Analysis: VietorisRipsPersistence + PersistenceLandscape
  Section A7.3  — Sector GCN embeddings (torch_geometric)
  Section A2.4  — RegimePersistencePosterior (Beta-Bernoulli conjugate updates)
  Section A10.4 — Population Stability Index (PSI) drift detector
"""

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

# ── Optional heavy dependencies with graceful fallbacks ───────────────────────
try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceLandscape
    GTDA_AVAILABLE = True
except ImportError:
    GTDA_AVAILABLE = False
    print("[feature_engineering] gtda not available — using spectral-norm TDA proxy")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    TORCH_GEO_AVAILABLE = True
except ImportError:
    TORCH_GEO_AVAILABLE = False
    print("[feature_engineering] torch_geometric not available — GCN features will be skipped")

# =============================================================================
# Section A4.5 — Core Regime Feature Engineering
# =============================================================================

def engineer_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix for Indian equity regime classification.

    Generates 30+ point-in-time features ensuring no look-ahead bias.
    All rolling windows use only past data (no `.shift(-1)` tricks).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Close, Midcap_Close, Smallcap_Close, IndiaVIX,
        Advances, Declines, NewHighs, NewLows, PctAbove50DMA, FII_Equity,
        DII_Equity, USDINR, Gilt10Y, AAA10Y, SIP_Monthly

    Returns
    -------
    pd.DataFrame  — feature matrix, NaN rows dropped
    """
    f = pd.DataFrame(index=df.index)

    # ── Returns ────────────────────────────────────────────────────────────────
    nifty_ret = df["Close"].pct_change()
    f["ret_1d"]  = nifty_ret
    f["ret_5d"]  = df["Close"].pct_change(5)
    f["ret_21d"] = df["Close"].pct_change(21)
    f["ret_63d"] = df["Close"].pct_change(63)

    # ── Trend ──────────────────────────────────────────────────────────────────
    ma50  = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()
    f["ma_50_200"]    = (ma50 / (ma200 + 1e-9)) - 1
    f["above_200dma"] = (df["Close"] > ma200).astype(int)
    f["trend_accel"]  = f["ma_50_200"].diff(10)  # rate of change of golden-cross

    # ── Volatility ─────────────────────────────────────────────────────────────
    f["vol_21d"]  = nifty_ret.rolling(21).std() * np.sqrt(252)
    f["vol_63d"]  = nifty_ret.rolling(63).std() * np.sqrt(252)
    f["vol_ratio"] = f["vol_21d"] / (f["vol_63d"] + 1e-9)
    f["parkinson"] = _parkinson_vol(df["Close"], window=21)  # high-low proxy

    # ── India VIX ─────────────────────────────────────────────────────────────
    f["vix_level"]    = df["IndiaVIX"]
    f["vix_change_5d"] = df["IndiaVIX"].pct_change(5)
    f["vix_z"]         = _zscore(df["IndiaVIX"], window=252)
    f["vix_vol_ratio"] = df["IndiaVIX"] / (f["vol_21d"] * 100 + 1e-9)  # term structure proxy

    # ── Breadth ───────────────────────────────────────────────────────────────
    f["adv_dec_ratio"]  = df["Advances"] / (df["Declines"] + 1e-9)
    f["pct_above_50dma"] = df["PctAbove50DMA"]
    f["new_highs_lows"] = df["NewHighs"] - df["NewLows"]
    f["mcclellan"]      = _mcclellan_oscillator(df["Advances"], df["Declines"])

    # ── Macro ─────────────────────────────────────────────────────────────────
    f["gilt_10y_change_21d"] = df["Gilt10Y"].diff(21)
    f["inr_change_21d"]      = df["USDINR"].pct_change(21)
    f["credit_spread"]       = df["AAA10Y"] - df["Gilt10Y"]
    f["credit_spread_z"]     = _zscore(f["credit_spread"], window=252)
    inr_ret_21 = df["USDINR"].pct_change(21)
    f["inr_stress_z"]        = _zscore(inr_ret_21, window=252)

    # ── Flows ─────────────────────────────────────────────────────────────────
    f["fii_eq_5d"]    = df["FII_Equity"].rolling(5).sum()
    f["dii_eq_5d"]    = df["DII_Equity"].rolling(5).sum()
    f["flow_balance"] = f["dii_eq_5d"] / (abs(f["fii_eq_5d"]) + 1e-9)
    f["fpi_z_60d"]    = _zscore(df["FII_Equity"], window=60)
    f["sip_momentum"] = df["SIP_Monthly"].pct_change(63)
    f["flow_divergence"] = (
        np.sign(df["DII_Equity"])
        * (df["DII_Equity"] > 0).astype(int)
        * (df["FII_Equity"] < 0).astype(int)
    )

    # ── Cap-Segmented ─────────────────────────────────────────────────────────
    f["midcap_rel_21d"]   = df["Midcap_Close"].pct_change(21) - f["ret_21d"]
    f["smallcap_rel_21d"] = df["Smallcap_Close"].pct_change(21) - f["ret_21d"]
    f["cap_divergence"]   = f["midcap_rel_21d"] - f["smallcap_rel_21d"]

    return f.dropna()


# =============================================================================
# Section A7.2 — Topological Data Analysis
# =============================================================================

def compute_tda_features(
    df: pd.DataFrame,
    window: int = 63,
    n_layers: int = 5,
    n_bins: int = 100,
) -> pd.DataFrame:
    """
    Extract persistence landscape features from rolling correlation matrices.

    Uses VietorisRipsPersistence on distance matrices derived from rolling
    Pearson correlations. Falls back to spectral-norm proxy if giotto-tda
    is unavailable.

    Parameters
    ----------
    df : pd.DataFrame  — price DataFrame (must have Close, Midcap_Close, Smallcap_Close,
                          IndiaVIX, Gilt10Y, USDINR, FII_Equity, DII_Equity)
    window : int        — rolling correlation window in business days
    n_layers, n_bins    — persistence landscape parameters (ignored in fallback)

    Returns
    -------
    pd.DataFrame indexed like df
    """
    price_cols = [c for c in ["Close", "Midcap_Close", "Smallcap_Close",
                                "IndiaVIX", "Gilt10Y", "USDINR"] if c in df.columns]
    rets = df[price_cols].pct_change()

    if GTDA_AVAILABLE:
        return _tda_full(rets, window, n_layers, n_bins)
    else:
        return _tda_proxy(rets, window)


def _tda_full(rets: pd.DataFrame, window: int, n_layers: int, n_bins: int) -> pd.DataFrame:
    """Full TDA pipeline using giotto-tda."""
    n = len(rets)
    feat_rows = []
    idx = []

    # Build rolling correlation distance tensors
    corr_cubes = []
    valid_dates = []
    for i in range(window, n):
        sub = rets.iloc[i - window: i].dropna(axis=1)
        if sub.shape[1] < 2:
            continue
        corr = sub.corr().fillna(0).values
        dist = np.sqrt(np.clip(2 * (1 - corr), 0, None))
        corr_cubes.append(dist)
        valid_dates.append(rets.index[i])

    if not corr_cubes:
        return pd.DataFrame(index=rets.index)

    corr_tensor = np.stack(corr_cubes)  # (T_valid, N, N)

    VR = VietorisRipsPersistence(
        metric="precomputed",
        homology_dimensions=[0, 1],
        n_jobs=-1,
    )
    diagrams = VR.fit_transform(corr_tensor)

    PL = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
    landscapes = PL.fit_transform(diagrams)
    flat = landscapes.reshape(len(landscapes), -1)

    feature_names = [f"tda_pl_{j}" for j in range(flat.shape[1])]
    tda_df = pd.DataFrame(flat, index=valid_dates, columns=feature_names)
    return tda_df.reindex(rets.index)


def _tda_proxy(rets: pd.DataFrame, window: int) -> pd.DataFrame:
    """Spectral-norm + trace proxy when giotto-tda is unavailable."""
    spectral_norms, traces, det_logs = [], [], []
    for i in range(len(rets)):
        if i < window:
            spectral_norms.append(np.nan)
            traces.append(np.nan)
            det_logs.append(np.nan)
        else:
            sub = rets.iloc[i - window: i].dropna(axis=1)
            if sub.shape[1] > 1:
                corr = sub.corr().fillna(0).values
                eigs = np.linalg.eigvalsh(corr)
                eigs_pos = np.clip(eigs, 1e-10, None)
                spectral_norms.append(float(eigs.max()))
                traces.append(float(np.trace(corr)))
                det_logs.append(float(np.log(eigs_pos).sum()))
            else:
                spectral_norms.append(1.0)
                traces.append(1.0)
                det_logs.append(0.0)

    return pd.DataFrame({
        "corr_spectral_norm": spectral_norms,
        "corr_trace": traces,
        "corr_log_det": det_logs,
    }, index=rets.index)


# =============================================================================
# Section A7.3 — Sector Graph Convolutional Network
# =============================================================================

class SectorGCN(nn.Module if TORCH_GEO_AVAILABLE else object):
    """
    Lightweight 2-layer GCN for sector-level regime feature extraction.
    Encodes the rolling correlation graph of sectors into a fixed-dim embedding.

    Input:  node features (N_sectors, in_dim) + edge_index
    Output: graph-level embedding (out_dim,) — one vector per day
    """

    def __init__(self, in_dim: int = 8, hidden: int = 64, out_dim: int = 32):
        if not TORCH_GEO_AVAILABLE:
            return
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.out_dim = out_dim

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        return x.mean(dim=0)  # graph-level pooling


def compute_gcn_embeddings(
    df: pd.DataFrame,
    window: int = 63,
    out_dim: int = 32,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Roll a correlation-threshold graph over time and pass through SectorGCN.

    Returns a DataFrame of shape (T, out_dim). Falls back to zeros if
    torch_geometric is unavailable.
    """
    price_cols = [c for c in ["Close", "Midcap_Close", "Smallcap_Close",
                                "Gilt10Y", "USDINR", "FII_Equity",
                                "DII_Equity", "IndiaVIX"] if c in df.columns]
    n_sectors = len(price_cols)
    rets = df[price_cols].pct_change()

    if not TORCH_GEO_AVAILABLE:
        cols = [f"gcn_{i}" for i in range(out_dim)]
        return pd.DataFrame(
            np.zeros((len(df), out_dim)), index=df.index, columns=cols
        )

    model = SectorGCN(in_dim=n_sectors, hidden=64, out_dim=out_dim)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in range(len(rets)):
            if i < window:
                embeddings.append(np.zeros(out_dim))
                continue
            sub = rets.iloc[i - window: i].fillna(0)
            node_feats = torch.tensor(sub.values.T.astype(np.float32))  # (N, window)
            # Use last-value features (N, n_sectors) → node feature = stat summaries
            node_x = torch.stack([
                node_feats.mean(1),
                node_feats.std(1),
                node_feats.min(1).values,
                node_feats.max(1).values,
            ], dim=1)  # (N, 4)

            # Build edges from correlation matrix
            corr = torch.tensor(sub.corr().fillna(0).values.astype(np.float32))
            mask = (corr.abs() > threshold) & (~torch.eye(n_sectors, dtype=torch.bool))
            edge_index = mask.nonzero().T.contiguous()
            edge_weight = corr[mask]

            # Pad node features to in_dim=8
            pad = torch.zeros(n_sectors, max(0, 8 - node_x.shape[1]))
            node_x_padded = torch.cat([node_x, pad], dim=1)[:, :8]

            if edge_index.shape[1] == 0:
                embeddings.append(np.zeros(out_dim))
            else:
                emb = model(node_x_padded, edge_index, edge_weight)
                embeddings.append(emb.numpy())

    cols = [f"gcn_{i}" for i in range(out_dim)]
    return pd.DataFrame(embeddings, index=df.index, columns=cols)


# =============================================================================
# Section A2.4 — Beta-Bernoulli Regime Persistence Posterior
# =============================================================================

class RegimePersistencePosterior:
    """
    Beta-Bernoulli model for regime self-transition probability.

    Tracks two conjugate Beta(alpha, beta) posteriors — one per regime —
    and updates them online as regime transitions are observed.

    Example
    -------
    >>> post = RegimePersistencePosterior()
    >>> post.update(persisted=55, transitioned=5)
    >>> print(f"P(stay) = {post.posterior_mean():.3f}")
    >>> print(f"95% CI  = {post.credible_interval()}")
    """

    def __init__(self, alpha: float = 2.0, beta: float = 1.0):
        self.alpha = alpha  # prior pseudo-counts of persistence
        self.beta  = beta   # prior pseudo-counts of transition

    def update(self, persisted: int, transitioned: int) -> "RegimePersistencePosterior":
        """Sequential Bayesian update from new regime observations."""
        self.alpha += persisted
        self.beta  += transitioned
        return self

    def update_from_sequence(self, regime_sequence: np.ndarray) -> "RegimePersistencePosterior":
        """Convenience: update from a full decoded regime sequence."""
        persisted    = int((regime_sequence[1:] == regime_sequence[:-1]).sum())
        transitioned = int((regime_sequence[1:] != regime_sequence[:-1]).sum())
        return self.update(persisted, transitioned)

    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        lo = (1 - level) / 2
        hi = 1 - lo
        return (
            float(beta_dist.ppf(lo, self.alpha, self.beta)),
            float(beta_dist.ppf(hi, self.alpha, self.beta)),
        )

    def summary(self) -> dict:
        ci = self.credible_interval()
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "posterior_mean": self.posterior_mean(),
            "ci_95_lo": ci[0],
            "ci_95_hi": ci[1],
        }


# =============================================================================
# Population Stability Index — Drift Detector
# =============================================================================

def population_stability_index(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute PSI between two distributions.

    PSI < 0.10  : No significant drift
    PSI < 0.25  : Moderate drift — monitor
    PSI >= 0.25 : Significant drift — trigger retraining

    Parameters
    ----------
    baseline : 1D array  — reference period distribution values
    current  : 1D array  — monitoring period distribution values
    n_bins   : int        — number of equal-width bins

    Returns
    -------
    float — PSI score
    """
    eps = 1e-6
    bins = np.linspace(
        min(baseline.min(), current.min()),
        max(baseline.max(), current.max()),
        n_bins + 1
    )
    base_counts, _ = np.histogram(baseline, bins=bins)
    curr_counts, _ = np.histogram(current, bins=bins)

    base_pct = base_counts / (base_counts.sum() + eps) + eps
    curr_pct = curr_counts / (curr_counts.sum() + eps) + eps

    psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return float(psi)


def compute_feature_psi(
    features: pd.DataFrame,
    train_end: str,
    n_bins: int = 10,
) -> pd.Series:
    """
    Compute PSI for each feature column between training and recent periods.

    Parameters
    ----------
    features  : feature DataFrame
    train_end : date string marking end of training period
    n_bins    : bins for PSI computation

    Returns
    -------
    pd.Series — PSI per feature column
    """
    baseline = features.loc[:train_end]
    current  = features.loc[train_end:]
    psi_scores = {}
    for col in features.columns:
        b = baseline[col].dropna().values
        c = current[col].dropna().values
        if len(b) > 5 and len(c) > 5:
            psi_scores[col] = population_stability_index(b, c, n_bins)
        else:
            psi_scores[col] = np.nan
    return pd.Series(psi_scores, name="PSI")


# =============================================================================
# Crisis Alert Composite
# =============================================================================

def covid_style_crash_alert(df: pd.DataFrame) -> pd.Categorical:
    """
    Composite crisis detector. Counts independent stress channels firing.
    Returns a Categorical: NORMAL / MONITOR / WARNING / ACUTE RISK-OFF.
    """
    vix_z     = _zscore(df["IndiaVIX"], 252)
    fii_z_60  = _zscore(df["FII_Equity"], 60)
    inr_z_60  = _zscore(df["USDINR"].pct_change(21), 252)
    breadth   = df["PctAbove50DMA"]

    score = (
        (vix_z > 2.5).astype(int)
        + (breadth < 25).astype(int)
        + (fii_z_60 < -1.5).astype(int)
        + (inr_z_60 > 1.5).astype(int)
    )
    return pd.cut(
        score,
        bins=[-1, 0, 1, 2, 4],
        labels=["NORMAL", "MONITOR", "WARNING", "ACUTE RISK-OFF"],
    )


def cap_segmented_stress(conviction_by_cap: dict, threshold: float = 0.4) -> str:
    """Flag when small-cap conviction collapses vs large-cap."""
    divergence = conviction_by_cap.get("Large Cap", 0.0) - conviction_by_cap.get("Small Cap", 0.0)
    return "CAP_DIVERGENCE_WARNING" if divergence > threshold else "ALIGNED"


def event_adjusted_conviction(base_conviction: float, days_to_event: int) -> float:
    """Halve conviction within 5 days of a known scheduled event."""
    near_event = 0 <= days_to_event < 5
    return base_conviction * 0.5 if near_event else base_conviction


# =============================================================================
# Private helpers
# =============================================================================

def _zscore(series: pd.Series, window: int) -> pd.Series:
    mu  = series.rolling(window).mean()
    sig = series.rolling(window).std()
    return (series - mu) / (sig + 1e-9)


def _mcclellan_oscillator(advances: pd.Series, declines: pd.Series) -> pd.Series:
    """19-day EMA minus 39-day EMA of advance-decline difference."""
    diff = advances - declines
    ema19 = diff.ewm(span=19, adjust=False).mean()
    ema39 = diff.ewm(span=39, adjust=False).mean()
    return ema19 - ema39


def _parkinson_vol(close: pd.Series, window: int = 21) -> pd.Series:
    """Parkinson estimator using high-low proxy (close-to-close rolling range)."""
    log_ret = np.log(close).diff()
    hl_proxy = log_ret.rolling(window).apply(lambda x: np.ptp(x), raw=True)
    return (hl_proxy / (np.sqrt(4 * np.log(2) * window) + 1e-9)) * np.sqrt(252)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.synthetic_data import generate_synthetic_market_data

    print("Generating market data...")
    df, regimes = generate_synthetic_market_data(seed=42)

    print("Engineering core features...")
    feats = engineer_regime_features(df)
    print(f"  Core features shape: {feats.shape}")

    print("Computing TDA features...")
    tda = compute_tda_features(df, window=63)
    tda_valid = tda.dropna()
    print(f"  TDA features shape: {tda_valid.shape}")

    print("Computing Beta-Bernoulli persistence posterior...")
    post = RegimePersistencePosterior()
    post.update_from_sequence(regimes)
    print(f"  Regime persistence posterior: {post.summary()}")

    print("Crash alert distribution:")
    alerts = covid_style_crash_alert(df)
    print(alerts.value_counts().to_string())

    print("\nDone.")
