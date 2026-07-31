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

if __name__ == "__main__":
    P = np.array([
        [0.90, 0.05, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.15, 0.80]
    ])
    mu = np.array([0.001, -0.002, 0.000])
    sigma = np.array([0.01, 0.025, 0.015])
    pf = RegimeParticleFilter(P, mu, sigma, n_particles=1000)
    obs_series = [0.0012, 0.0008, -0.015, -0.022, 0.0005]
    for obs in obs_series:
        post = pf.step(obs)
        print(f"Obs: {obs:+.4f} | Posterior state probabilities: {np.round(post, 3)}")

