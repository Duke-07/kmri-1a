import numpy as np

class RegimeParticleFilter:
    """bootstrap particle filter for a k-regime HMM with Gaussian emissions."""
    def __init__(self, P, mu, sigma, n_particles=5000, seed=42):
        self.P = P # (k, k) transition matrix
        self.mu = mu # (k,) regime means
        self.sigma = sigma # (k,) regime stds
        self.K = P.shape[0]
        self.N = n_particles
        self.rng = np.random.default_rng(seed)
        self.particles = self.rng.integers(0, self.K, size=self.N)
        self.weights = np.full(self.N, 1.0 / self.N)
        
    def step(self, obs):
        # 1) propagate each particle through the transition matrix
        self.particles = np.array([
            self.rng.choice(self.K, p=self.P[s]) for s in self.particles
        ])
        
        # 2) re-weight by Gaussian emission likelihood
        like = np.exp(-0.5 * ((obs - self.mu[self.particles]) / self.sigma[self.particles]) ** 2) / (self.sigma[self.particles] * np.sqrt(2 * np.pi))
        self.weights *= like
        if self.weights.sum() > 0:
            self.weights /= self.weights.sum()
        else:
            self.weights = np.full(self.N, 1.0 / self.N)
            
        # 3) systematic resample if effective sample size is low
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.N / 2:
            idx = self.rng.choice(self.N, size=self.N, p=self.weights)
            self.particles = self.particles[idx]
            self.weights = np.full(self.N, 1.0 / self.N)
            
        # regime posterior at this step
        post = np.bincount(self.particles, weights=self.weights, minlength=self.K)
        return post / post.sum()
