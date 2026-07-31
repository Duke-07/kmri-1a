import pymc as pm
import pytensor.tensor as pt
import numpy as np

def build_bayesian_rsvar(Y: np.ndarray, K=5):
    """
    Bayesian regime-switching VAR(1). 
    Y: (T, d) feature matrix.
    """
    T, d = Y.shape
    with pm.Model() as model:
        # Persistence-favouring Dirichlet rows for the transition matrix
        alpha = np.eye(K) * 8.0 + (1 - np.eye(K)) * 1.0
        P = pm.Dirichlet('P', a=alpha, shape=(K, K))
        pi = pm.Dirichlet('pi', a=np.ones(K), shape=K)
        
        # Regime-conditional intercepts and VAR(1) coefficient matrices
        c = pm.Normal('c', 0.0, 0.05, shape=(K, d))
        Avar = pm.Normal('A', 0.0, 0.3, shape=(K, d, d))
        
        # Regime-conditional covariance via LKJ + half-normal scales
        chol, corr, sd = pm.LKJCholeskyCov(
            'chol', n=d, eta=2.0,
            sd_dist=pm.HalfNormal.dist(0.03, shape=d), compute_corr=True
        )
        
        # Note: True marginalization of Hidden Markov Chain may require a custom distribution
        # or pm.HiddenMarkovChain (if available in pymc-experimental). 
        # Using Categorical for simplicity in this baseline structural representation.
        states = pm.Categorical('states', p=pi, shape=T)
        
        mu = c[states[1:]] + pt.batched_dot(Avar[states[1:]], Y[:-1])
        pm.MvNormal('obs', mu=mu, chol=chol[states[1:]], observed=Y[1:])
        
    return model

if __name__ == "__main__":
    print("RS-VAR module loaded.")
    # Ex:
    # Y = np.random.normal(0, 0.01, (100, 3))
    # model = build_bayesian_rsvar(Y, K=3)
