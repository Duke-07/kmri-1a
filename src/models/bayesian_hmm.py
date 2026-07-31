import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd

def build_bayesian_hmm(returns: np.ndarray, K=5):
    """
    Bayesian Gaussian-emission HMM with Dirichlet transition rows.
    """
    T = len(returns)
    with pm.Model() as model:
        # Priors for transition matrix
        alpha_diag = 8.0
        alpha_off = 1.0
        alpha_mat = np.eye(K) * alpha_diag + (1 - np.eye(K)) * alpha_off
        
        P = pm.Dirichlet('P', a=alpha_mat, shape=(K, K))
        pi = pm.Dirichlet('pi', a=np.ones(K), shape=K)
        
        # Emission parameters
        mu = pm.Normal('mu', mu=0, sigma=0.02, shape=K)
        sigma = pm.HalfNormal('sigma', sigma=0.03, shape=K)
        
        # Latent states
        states = pm.Categorical('states', p=pi, shape=T)
        
        # Observations
        obs = pm.Normal('obs', mu=mu[states], sigma=sigma[states], observed=returns)
        
    return model

def sample_bayesian_hmm(model, draws=1000, tune=500, chains=2, seed=42):
    with model:
        trace = pm.sample(draws=draws, tune=tune, target_accept=0.95, random_seed=seed, chains=chains)
    return trace

if __name__ == "__main__":
    print("Bayesian HMM module loaded.")
    # Ex:
    # rets = np.random.normal(0, 0.01, 100)
    # model = build_bayesian_hmm(rets)
    # trace = sample_bayesian_hmm(model, draws=100, tune=100, chains=1)
