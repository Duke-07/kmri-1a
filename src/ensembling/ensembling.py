import numpy as np
from scipy.optimize import minimize
import pandas as pd

def bma_weights(log_predictive_likelihoods):
    """
    log_predictive_likelihoods: (m,) out-of-sample log-lik per model.
    """
    w = np.exp(log_predictive_likelihoods - log_predictive_likelihoods.max())
    return w / w.sum()

def bma_combine(model_probs, weights):
    """
    model_probs: (m, k) regime probs per model; weights: (m,).
    """
    return np.tensordot(weights, model_probs, axes=(0, 0)) # (k,)

def fit_stacking_weights(base_probs, y_true):
    """
    base_probs: (m, n, k) out-of-fold probs; y_true: (n,) labels.
    returns simplex weights minimising mean cross-entropy.
    """
    M, N, K = base_probs.shape
    onehot = np.eye(K)[y_true]
    
    def neg_loglik(w):
        w = np.clip(w, 0, None)
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.ones(M) / M
        combined = np.tensordot(w, base_probs, axes=(0, 0)) # (n, k)
        combined = np.clip(combined, 1e-9, 1.0)
        return -np.mean(np.sum(onehot * np.log(combined), axis=1))
        
    w0 = np.full(M, 1.0 / M)
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1},)
    bnds = [(0, 1)] * M
    res = minimize(neg_loglik, w0, bounds=bnds, constraints=cons)
    return res.x / res.x.sum()

if __name__ == "__main__":
    # 3 models, 100 samples, 5 regime classes
    np.random.seed(42)
    m1_probs = np.random.dirichlet(np.ones(5), size=100)
    m2_probs = np.random.dirichlet(np.ones(5), size=100)
    m3_probs = np.random.dirichlet(np.ones(5), size=100)
    base_probs = np.stack([m1_probs, m2_probs, m3_probs])
    y_true = np.random.randint(0, 5, size=100)
    
    weights = fit_stacking_weights(base_probs, y_true)
    print("Fitted Stacking Weights for 3 ensemble models:", np.round(weights, 4))
    
    log_liks = np.array([-1.2, -1.0, -1.5])
    bma_w = bma_weights(log_liks)
    print("BMA Weights for models:", np.round(bma_w, 4))

