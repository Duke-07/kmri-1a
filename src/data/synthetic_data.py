import pandas as pd
import numpy as np
import os

def generate_synthetic_market_data(start_date='2007-01-01', end_date='2024-12-31', seed=42):
    """
    Generates synthetic daily market data mimicking Indian equity markets.
    Contains Nifty 50, Midcap, Smallcap, VIX, FII/DII flows, macro indicators.
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq='B') # Business days
    n_days = len(dates)
    
    # Simulate true hidden regimes (0: Risk-On, 1: Risk-Off, 2: Transitional)
    P = np.array([
        [0.98, 0.01, 0.01],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90]
    ])
    
    regimes = np.zeros(n_days, dtype=int)
    regimes[0] = 0
    for t in range(1, n_days):
        regimes[t] = np.random.choice(3, p=P[regimes[t-1]])
        
    means = {0: 0.0005, 1: -0.0010, 2: 0.0000}
    vols = {0: 0.01, 1: 0.025, 2: 0.015}
    
    returns = np.zeros(n_days)
    for t in range(n_days):
        returns[t] = np.random.normal(means[regimes[t]], vols[regimes[t]])
        
    nifty_close = 1000 * np.exp(np.cumsum(returns))
    
    midcap_returns = returns * 1.2 + np.random.normal(0, 0.01, n_days)
    smallcap_returns = returns * 1.5 + np.random.normal(0, 0.015, n_days)
    
    vix_base = {0: 12.0, 1: 25.0, 2: 18.0}
    vix = np.zeros(n_days)
    for t in range(n_days):
        vix[t] = np.random.normal(vix_base[regimes[t]], vix_base[regimes[t]] * 0.1)
    vix = np.clip(vix, 10, 85)
    
    advances = np.zeros(n_days)
    for t in range(n_days):
        if returns[t] > 0:
            advances[t] = np.random.randint(1000, 1500)
        else:
            advances[t] = np.random.randint(300, 800)
    declines = 2000 - advances
    
    fii_flows = np.zeros(n_days)
    dii_flows = np.zeros(n_days)
    for t in range(n_days):
        if regimes[t] == 1:
            fii_flows[t] = np.random.normal(-2000, 1000)
            dii_flows[t] = np.random.normal(1500, 800)
        else:
            fii_flows[t] = np.random.normal(500, 1000)
            dii_flows[t] = np.random.normal(200, 500)
            
    usd_inr = 45.0 * np.exp(np.cumsum(np.random.normal(0.0001, 0.002, n_days)))
    gilt_10y = 7.0 + np.cumsum(np.random.normal(0, 0.05, n_days))
    aaa_10y = gilt_10y + np.random.normal(1.0, 0.2, n_days)
    
    new_highs = advances * 0.05 + np.random.normal(0, 10, n_days)
    new_lows = declines * 0.05 + np.random.normal(0, 10, n_days)
    
    pct_above_50dma = 50 + 20 * np.sin(np.linspace(0, 100, n_days)) + np.random.normal(0, 5, n_days)
    sip_monthly = np.linspace(3000, 20000, n_days) + np.random.normal(0, 500, n_days)
    
    df = pd.DataFrame({
        'Date': dates,
        'Close': nifty_close,
        'Midcap_Close': 1000 * np.exp(np.cumsum(midcap_returns)),
        'Smallcap_Close': 1000 * np.exp(np.cumsum(smallcap_returns)),
        'IndiaVIX': vix,
        'Advances': advances,
        'Declines': declines,
        'NewHighs': new_highs,
        'NewLows': new_lows,
        'PctAbove50DMA': np.clip(pct_above_50dma, 0, 100),
        'FII_Equity': fii_flows,
        'DII_Equity': dii_flows,
        'USDINR': usd_inr,
        'Gilt10Y': gilt_10y,
        'AAA10Y': aaa_10y,
        'SIP_Monthly': sip_monthly
    })
    
    df.set_index('Date', inplace=True)
    return df, regimes

if __name__ == "__main__":
    df, hidden_regimes = generate_synthetic_market_data()
    df.to_csv(os.path.join(os.path.dirname(__file__), "synthetic_indian_market.csv"))
    print("Generated synthetic_indian_market.csv")
