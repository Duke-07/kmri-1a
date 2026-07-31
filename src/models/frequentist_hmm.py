import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

def fit_regime_hmm(returns: pd.Series, n_states=5, n_iter=200, seed=42):
    """fit a Gaussian-emission HMM to a returns series."""
    X = returns.values.reshape(-1, 1)
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=seed,
        tol=1e-5
    )
    model.fit(X)
    states = model.predict(X)
    state_probs = model.predict_proba(X)
    return model, states, state_probs

def label_regimes(model, K=5):
    """map numeric states to economic regime names by mean/vol signature."""
    summary = []
    for i in range(K):
        mu = model.means_[i, 0]
        sig = np.sqrt(model.covars_[i, 0, 0])
        summary.append((i, mu, sig))
    
    # sort by (mean, -vol): risk-on = high mean low vol; risk-off = low mean high vol
    summary.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    labels = ['Risk-On', 'Late-Cycle', 'Transitional', 'Post-Shock', 'Risk-Off']
    
    # for k != 5, fallback
    if K != 5:
        labels = [f"Regime_{j}" for j in range(K)]
        
    mapping = {summary[r][0]: labels[r] for r in range(K)}
    return mapping

if __name__ == "__main__":
    import os
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_indian_market.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        df['Return'] = df['Close'].pct_change()
        model, states, probs = fit_regime_hmm(df['Return'].dropna(), n_states=5)
        mapping = label_regimes(model)
        print("Regime Mapping:", mapping)
        print("Transition Matrix:\n", model.transmat_)
