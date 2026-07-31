import pandas as pd
import numpy as np

def engineer_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """build feature matrix for indian equity regime classification."""
    f = pd.DataFrame(index=df.index)
    
    # return features
    f['ret_1d'] = df['Close'].pct_change()
    f['ret_5d'] = df['Close'].pct_change(5)
    f['ret_21d'] = df['Close'].pct_change(21)
    f['ret_63d'] = df['Close'].pct_change(63)
    
    # trend features
    f['ma_50_200'] = (df['Close'].rolling(50).mean() / df['Close'].rolling(200).mean() - 1)
    f['above_200dma'] = (df['Close'] > df['Close'].rolling(200).mean()).astype(int)
    
    # volatility features
    f['vol_21d'] = df['Close'].pct_change().rolling(21).std() * np.sqrt(252)
    f['vol_63d'] = df['Close'].pct_change().rolling(63).std() * np.sqrt(252)
    f['vol_ratio'] = f['vol_21d'] / (f['vol_63d'] + 1e-9)
    
    # india VIX features
    f['vix_level'] = df['IndiaVIX']
    f['vix_change_5d'] = df['IndiaVIX'].pct_change(5)
    f['vix_z'] = (df['IndiaVIX'] - df['IndiaVIX'].rolling(252).mean()) / (df['IndiaVIX'].rolling(252).std() + 1e-9)
    
    # breadth features
    f['adv_dec_ratio'] = df['Advances'] / (df['Declines'] + 1e-9)
    f['pct_above_50dma'] = df['PctAbove50DMA']
    f['new_highs_lows'] = df['NewHighs'] - df['NewLows']
    
    # macro features
    f['gilt_10y_change_21d'] = df['Gilt10Y'].diff(21)
    f['inr_change_21d'] = df['USDINR'].pct_change(21)
    f['credit_spread'] = df['AAA10Y'] - df['Gilt10Y']
    
    # flow features
    f['fii_eq_5d'] = df['FII_Equity'].rolling(5).sum()
    f['dii_eq_5d'] = df['DII_Equity'].rolling(5).sum()
    f['flow_balance'] = f['dii_eq_5d'] / (abs(f['fii_eq_5d']) + 1e-9)
    
    # additional cap-segmented features (section C2)
    f['midcap_rel_perf_21d'] = df['Midcap_Close'].pct_change(21) - f['ret_21d']
    f['smallcap_rel_perf_21d'] = df['Smallcap_Close'].pct_change(21) - f['ret_21d']
    
    # additional flow features (section A24)
    f['fpi_z_60d'] = (df['FII_Equity'] - df['FII_Equity'].rolling(60).mean()) / (df['FII_Equity'].rolling(60).std() + 1e-9)
    f['sip_momentum'] = df['SIP_Monthly'].pct_change(63) # ~3-month trend
    f['flow_divergence'] = np.sign(df['DII_Equity']) * (df['DII_Equity'] > 0) * (df['FII_Equity'] < 0)
    
    return f.dropna()

def compute_topology_features(df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """
    Extract topological persistence features (Section A7.2) from rolling correlation matrices.
    If giotto-tda is unavailable, computes spectral norm and trace as topology proxies.
    """
    tda_features = pd.DataFrame(index=df.index)
    returns = df.pct_change()
    
    spectral_norms = []
    traces = []
    for i in range(len(df)):
        if i < window:
            spectral_norms.append(np.nan)
            traces.append(np.nan)
        else:
            sub = returns.iloc[i-window:i].dropna(axis=1)
            if sub.shape[1] > 1:
                corr = sub.corr().fillna(0).values
                eigs = np.linalg.eigvalsh(corr)
                spectral_norms.append(float(np.max(eigs)))
                traces.append(float(np.trace(corr)))
            else:
                spectral_norms.append(1.0)
                traces.append(1.0)
                
    tda_features['corr_spectral_norm'] = spectral_norms
    tda_features['corr_trace'] = traces
    return tda_features

def covid_style_crash_alert(df: pd.DataFrame) -> pd.Series:
    """
    Composite crisis detector (Section C1.4).
    Counts how many independent stress channels fire on a given day.
    """
    vix_z = (df['IndiaVIX'] - df['IndiaVIX'].rolling(252).mean()) / (df['IndiaVIX'].rolling(252).std() + 1e-9)
    pct_above_50dma = df['PctAbove50DMA']
    fii_flow_z_60d = (df['FII_Equity'] - df['FII_Equity'].rolling(60).mean()) / (df['FII_Equity'].rolling(60).std() + 1e-9)
    usdinr_z_60d = (df['USDINR'].pct_change(21) - df['USDINR'].pct_change(21).rolling(252).mean()) / (df['USDINR'].pct_change(21).rolling(252).std() + 1e-9)
    
    vix_spike = (vix_z > 2.5).astype(int)
    breadth_collapse = (pct_above_50dma < 25).astype(int)
    fii_outflow = (fii_flow_z_60d < -1.5).astype(int)
    inr_stress = (usdinr_z_60d > 1.5).astype(int)
    
    signal_count = vix_spike + breadth_collapse + fii_outflow + inr_stress
    return pd.cut(signal_count, bins=[-1, 0, 1, 2, 4], labels=['NORMAL', 'MONITOR', 'WARNING', 'ACUTE RISK-OFF'])

def cap_segmented_stress(conviction_by_cap: dict, threshold: float = 0.4) -> str:
    """
    Capitalisation-divergence detector (Section C2.3).
    Flags when small-cap conviction collapses while large-cap conviction holds.
    """
    divergence = conviction_by_cap.get('Large Cap', 0.0) - conviction_by_cap.get('Small Cap', 0.0)
    return 'CAP_DIVERGENCE_WARNING' if divergence > threshold else 'ALIGNED'

def event_adjusted_conviction(base_conviction: float, days_to_event: int) -> float:
    """
    Calendar-aware regime conviction adjustment (Section C5.3).
    Halves conviction within 5 days of a known scheduled event.
    """
    near_event = 0 <= days_to_event < 5
    return base_conviction * 0.5 if near_event else base_conviction

if __name__ == "__main__":
    import os
    file_path = os.path.join(os.path.dirname(__file__), "synthetic_indian_market.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        features = engineer_regime_features(df)
        print(f"Engineered {features.shape[1]} features for {features.shape[0]} days.")
        features.to_csv(os.path.join(os.path.dirname(__file__), "features.csv"))

