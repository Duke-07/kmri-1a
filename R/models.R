# R Codebase for Baseline Models and Cross-Validation
# Requires: depmixS4, MSwM

library(depmixS4)
library(MSwM)

fit_hmm_baseline <- function(returns, k = 5) {
  # Fits a Gaussian HMM using depmixS4
  mod <- depmix(returns ~ 1, family = gaussian(), nstates = k, data = data.frame(returns))
  fit_mod <- fit(mod)
  post <- posterior(fit_mod)
  return(list(model = fit_mod, posterior = post))
}

fit_mswm_baseline <- function(returns, k = 3) {
  # Fits a Markov-Switching variance regression model
  mod_lm <- lm(returns ~ 1)
  msm <- msmFit(mod_lm, k = k, sw = c(TRUE, TRUE))
  return(msm)
}

# Example Usage
# returns <- diff(log(read.csv("../src/data/synthetic_indian_market.csv")$Close))
# hmm_res <- fit_hmm_baseline(returns)
# print(summary(hmm_res$model))
