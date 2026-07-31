import pandas as pd
import numpy as np

def engineer_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for Indian equity regime classification."""
    f = pd.DataFrame(index=df.index)
    
    # Return features
    f['ret_1d'] = df['Close'].pct_change()
    f['ret_5d'] = df['Close'].pct_change(5)
    f['ret_21d'] = df['Close'].pct_change(21)
    f['ret_63d'] = df['Close'].pct_change(63)
    
    # Trend features
    f['ma_50_200'] = (df['Close'].rolling(50).mean() / df['Close'].rolling(200).mean() - 1)
    f['above_200dma'] = (df['Close'] > df['Close'].rolling(200).mean()).astype(int)
    
    # Volatility features
    f['vol_21d'] = df['Close'].pct_change().rolling(21).std() * np.sqrt(252)
    f['vol_63d'] = df['Close'].pct_change().rolling(63).std() * np.sqrt(252)
    f['vol_ratio'] = f['vol_21d'] / (f['vol_63d'] + 1e-9)
    
    # India VIX features
    f['vix_level'] = df['IndiaVIX']
    f['vix_change_5d'] = df['IndiaVIX'].pct_change(5)
    f['vix_z'] = (df['IndiaVIX'] - df['IndiaVIX'].rolling(252).mean()) / (df['IndiaVIX'].rolling(252).std() + 1e-9)
    
    # Breadth features
    f['adv_dec_ratio'] = df['Advances'] / (df['Declines'] + 1e-9)
    f['pct_above_50dma'] = df['PctAbove50DMA']
    f['new_highs_lows'] = df['NewHighs'] - df['NewLows']
    
    # Macro features
    f['gilt_10y_change_21d'] = df['Gilt10Y'].diff(21)
    f['inr_change_21d'] = df['USDINR'].pct_change(21)
    f['credit_spread'] = df['AAA10Y'] - df['Gilt10Y']
    
    # Flow features
    f['fii_eq_5d'] = df['FII_Equity'].rolling(5).sum()
    f['dii_eq_5d'] = df['DII_Equity'].rolling(5).sum()
    f['flow_balance'] = f['dii_eq_5d'] / (abs(f['fii_eq_5d']) + 1e-9)
    
    # Additional Cap-segmented features (Section C2)
    f['midcap_rel_perf_21d'] = df['Midcap_Close'].pct_change(21) - f['ret_21d']
    f['smallcap_rel_perf_21d'] = df['Smallcap_Close'].pct_change(21) - f['ret_21d']
    
    # Additional flow features (Section A24)
    f['fpi_z_60d'] = (df['FII_Equity'] - df['FII_Equity'].rolling(60).mean()) / (df['FII_Equity'].rolling(60).std() + 1e-9)
    f['sip_momentum'] = df['SIP_Monthly'].pct_change(63) # ~3-month trend
    f['flow_divergence'] = np.sign(df['DII_Equity']) * (df['DII_Equity'] > 0) * (df['FII_Equity'] < 0)
    
    return f.dropna()

if __name__ == "__main__":
    import os
    file_path = os.path.join(os.path.dirname(__file__), "synthetic_indian_market.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        features = engineer_regime_features(df)
        print(f"Engineered {features.shape[1]} features for {features.shape[0]} days.")
        features.to_csv(os.path.join(os.path.dirname(__file__), "features.csv"))
