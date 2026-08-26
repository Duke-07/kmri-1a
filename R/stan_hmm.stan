// Stan HMM — Bayesian Regime Detection Engine
// Aaryan Dwivedi
// Exact forward-algorithm marginalisation over latent regime states.
// Run via CmdStanR: mod <- cmdstan_model("R/stan_hmm.stan"); fit <- mod$sample(...)

data {
  int<lower=1> T;         // number of time steps
  int<lower=1> K;         // number of regimes (5 for this project)
  vector[T] returns;       // observed daily returns
}

parameters {
  // Transition matrix rows: each row is a Dirichlet
  simplex[K] transition[K];
  
  // Initial state distribution
  simplex[K] pi;
  
  // Regime-conditional emission parameters
  vector[K] mu;              // regime means (daily return)
  vector<lower=0>[K] sigma;  // regime standard deviations
}

model {
  // ── Priors ─────────────────────────────────────────────────────────────────
  // Persistence-favouring Dirichlet: diagonal 8, off-diagonal 1
  for (k in 1:K) {
    vector[K] alpha_k;
    for (j in 1:K) {
      alpha_k[j] = (k == j) ? 8.0 : 1.0;
    }
    transition[k] ~ dirichlet(alpha_k);
  }
  
  pi ~ dirichlet(rep_vector(1.0, K));
  
  mu    ~ normal(0, 0.02);          // small daily return means
  sigma ~ normal(0, 0.03);          // positive via half-normal
  
  // ── Forward Algorithm (marginalise over hidden states) ────────────────────
  // log_alpha[t, k] = log P(y_1,...,y_t, S_t = k)
  {
    matrix[T, K] log_alpha;
    
    // t = 1: initialise
    for (k in 1:K) {
      log_alpha[1, k] = log(pi[k]) + normal_lpdf(returns[1] | mu[k], sigma[k]);
    }
    
    // t = 2,...,T: forward recursion
    for (t in 2:T) {
      for (k in 1:K) {
        vector[K] acc;
        for (j in 1:K) {
          acc[j] = log_alpha[t-1, j] + log(transition[j][k]);
        }
        log_alpha[t, k] = log_sum_exp(acc) + normal_lpdf(returns[t] | mu[k], sigma[k]);
      }
    }
    
    // Marginal likelihood = log sum_k P(y_{1:T}, S_T = k)
    target += log_sum_exp(log_alpha[T]);
  }
}

generated quantities {
  // Viterbi decoding (approximate — deterministic argmax forward pass)
  int states[T];
  {
    matrix[T, K] log_delta;
    int psi[T, K];
    
    for (k in 1:K) {
      log_delta[1, k] = log(pi[k]) + normal_lpdf(returns[1] | mu[k], sigma[k]);
      psi[1, k] = 0;
    }
    
    for (t in 2:T) {
      for (k in 1:K) {
        vector[K] scores;
        for (j in 1:K) {
          scores[j] = log_delta[t-1, j] + log(transition[j][k]);
        }
        log_delta[t, k] = max(scores) + normal_lpdf(returns[t] | mu[k], sigma[k]);
        psi[t, k] = sort_indices_desc(scores)[1];
      }
    }
    
    // Backtrack
    {
      int best_last;
      vector[K] last_scores;
      for (k in 1:K) last_scores[k] = log_delta[T, k];
      best_last = sort_indices_desc(last_scores)[1];
      states[T] = best_last;
      for (t in T-1:-1:1) {
        states[t] = psi[t+1, states[t+1]];
      }
    }
  }
}
