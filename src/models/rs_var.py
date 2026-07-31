import pymc as pm
import pytensor.tensor as pt
import numpy as np

def build_bayesian_rsvar(Y: np.ndarray, K=5):
    """
    bayesian regime-switching VAR(1). 
    y: (t, d) feature matrix.
    """
    T, d = Y.shape
    with pm.Model() as model:
        # persistence-favouring Dirichlet rows for the transition matrix
        alpha = np.eye(K) * 8.0 + (1 - np.eye(K)) * 1.0
        P = pm.Dirichlet('P', a=alpha, shape=(K, K))
        pi = pm.Dirichlet('pi', a=np.ones(K), shape=K)
        
        # regime-conditional intercepts and VAR(1) coefficient matrices
        c = pm.Normal('c', 0.0, 0.05, shape=(K, d))
        Avar = pm.Normal('A', 0.0, 0.3, shape=(K, d, d))
        
        # regime-conditional covariance via LKJ + half-Normal scales
        chol, corr, sd = pm.LKJCholeskyCov(
            'chol', n=d, eta=2.0,
            sd_dist=pm.HalfNormal.dist(0.03, shape=d), compute_corr=True
        )
        
        # note: true marginalization of hidden markov chain may require a custom distribution
        # or pm.hiddenmarkovchain (if available in PyMC-experimental). 
        # using Categorical for simplicity in this baseline structural representation.
        states = pm.Categorical('states', p=pi, shape=T)
        
        mu = c[states[1:]] + pt.batched_dot(Avar[states[1:]], Y[:-1])
        pm.MvNormal('obs', mu=mu, chol=chol[states[1:]], observed=Y[1:])
        
    return model

if __name__ == "__main__":
    print("RS-VAR module loaded.")
    # ex:
    # y = np.random.Normal(0, 0.01, (100, 3))
    # model = build_bayesian_rsvar(y, k=3)
