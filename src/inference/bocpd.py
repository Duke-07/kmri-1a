import numpy as np
from scipy.stats import norm

def bocpd(data, hazard=1/100, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
    """
    bayesian online changepoint detection with a Normal-gamma model.
    returns the run-length posterior matrix r (t+1, t+1).
    """
    T = len(data)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0
    
    mu, kappa, alpha, beta = [mu0], [kappa0], [alpha0], [beta0]
    
    for t, x in enumerate(data):
        # predictive prob of x under each run length (student-t)
        scale = np.sqrt(np.array(beta) * (np.array(kappa) + 1) / (np.array(alpha) * np.array(kappa)))
        pred = norm.pdf(x, loc=np.array(mu), scale=scale)
        
        R[1:t+2, t+1] = R[0:t+1, t] * pred * (1 - hazard) # growth
        R[0, t+1] = np.sum(R[0:t+1, t] * pred * hazard) # changepoint
        R[:, t+1] /= R[:, t+1].sum()
        
        # update sufficient statistics (Normal-gamma conjugacy)
        mu_new = (np.array(kappa) * np.array(mu) + x) / (np.array(kappa) + 1)
        kappa_new = np.array(kappa) + 1
        alpha_new = np.array(alpha) + 0.5
        beta_new = np.array(beta) + (np.array(kappa) * (x - np.array(mu))**2) / (2 * (np.array(kappa) + 1))
        
        mu = np.concatenate([[mu0], mu_new])
        kappa = np.concatenate([[kappa0], kappa_new])
        alpha = np.concatenate([[alpha0], alpha_new])
        beta = np.concatenate([[beta0], beta_new])
        
    return R
