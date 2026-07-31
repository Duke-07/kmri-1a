# r codebase for baseline models and cross-validation
# requires: depmixs4, mswm

library(depmixS4)
library(MSwM)

fit_hmm_baseline <- function(returns, k = 5) {
  # fits a Gaussian HMM using depmixs4
  mod <- depmix(returns ~ 1, family = gaussian(), nstates = k, data = data.frame(returns))
  fit_mod <- fit(mod)
  post <- posterior(fit_mod)
  return(list(model = fit_mod, posterior = post))
}

fit_mswm_baseline <- function(returns, k = 3) {
  # fits a markov-switching variance regression model
  mod_lm <- lm(returns ~ 1)
  msm <- msmFit(mod_lm, k = k, sw = c(TRUE, TRUE))
  return(msm)
}

# example usage
# returns <- diff(log(read.csv("../src/data/synthetic_indian_market.csv")$close))
# hmm_res <- fit_hmm_baseline(returns)
# print(summary(hmm_res$model))
