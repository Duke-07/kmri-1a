# =============================================================================
# R Codebase — Bayesian Regime Detection Engine
# =============================================================================
# Implements Section A3.6, Deliverable 3:
#   - Frequentist HMM via depmixS4 (5-state Gaussian)
#   - Markov-switching regression via MSwM (3 regimes)
#   - Bayesian regime model via rstanarm / brms
#   - Changepoint detection via bcp and changepoint packages
#   - Conformal prediction (manual implementation)
#   - Cross-validation reconciliation vs Python engine
#
# Requirements:
#   install.packages(c("depmixS4","MSwM","rstanarm","brms","bcp",
#                      "changepoint","tidyverse","PerformanceAnalytics"))
# =============================================================================

suppressPackageStartupMessages({
  library(depmixS4)
  library(MSwM)
  library(tidyverse)
})

# Conditionally load Bayesian / changepoint packages
safe_require <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("[R] Package '%s' not installed — some functions will be skipped.", pkg))
    return(FALSE)
  }
  library(pkg, character.only = TRUE)
  return(TRUE)
}

has_bcp        <- safe_require("bcp")
has_changepoint<- safe_require("changepoint")
has_rstanarm   <- safe_require("rstanarm")
has_brms       <- safe_require("brms")
has_perf       <- safe_require("PerformanceAnalytics")


# =============================================================================
# 1. Frequentist HMM — depmixS4 (Section A3.6)
# =============================================================================

#' Fit a K-state Gaussian HMM using depmixS4.
#'
#' @param returns  numeric vector of daily log-returns
#' @param k        number of hidden states (default 5)
#' @return         list(model, posterior, transition_matrix)
fit_hmm_depmixs4 <- function(returns, k = 5) {
  df  <- data.frame(returns = returns)
  mod <- depmix(returns ~ 1, family = gaussian(), nstates = k, data = df)
  fit <- tryCatch(fit(mod, verbose = FALSE), error = function(e) {
    message("[R] depmixS4 fit failed: ", e$message)
    NULL
  })
  if (is.null(fit)) return(NULL)
  post <- posterior(fit)
  
  # Extract transition matrix
  trans_mat <- getpars(fit)
  message(sprintf("[R] Fitted %d-state HMM | Log-likelihood: %.2f", k, logLik(fit)))
  
  list(
    model       = fit,
    posterior   = post,
    log_lik     = as.numeric(logLik(fit)),
    aic         = AIC(fit),
    bic         = BIC(fit),
    states      = post$state,
    probs       = post[, 2:(k + 1)]
  )
}


#' Compare HMMs by BIC across multiple K values.
#'
#' @param returns  numeric vector
#' @param k_values integer vector of state counts
compare_hmm_bic <- function(returns, k_values = c(3, 5, 7)) {
  results <- lapply(k_values, function(k) {
    res <- fit_hmm_depmixs4(returns, k = k)
    if (is.null(res)) return(data.frame(K=k, LogLik=NA, AIC=NA, BIC=NA))
    data.frame(K = k, LogLik = res$log_lik, AIC = res$aic, BIC = res$bic)
  })
  df <- do.call(rbind, results)
  df$BIC_rank <- rank(df$BIC)
  df
}


#' Compute regime duration statistics from a state sequence.
#'
#' @param states  integer vector of decoded regime labels
#' @param K       number of regimes
regime_durations <- function(states, K = 5) {
  durations <- lapply(1:K, function(k) {
    runs <- rle(states == k)
    run_lengths <- runs$lengths[runs$values == TRUE]
    if (length(run_lengths) == 0) return(data.frame(Regime=k, Count=0, Mean=NA, Max=NA))
    data.frame(
      Regime       = k,
      Count        = sum(run_lengths),
      N_Episodes   = length(run_lengths),
      Mean_Duration = round(mean(run_lengths), 1),
      Max_Duration  = max(run_lengths)
    )
  })
  do.call(rbind, durations)
}


# =============================================================================
# 2. Markov-Switching Variance Regression — MSwM (Section A8.2)
# =============================================================================

#' Fit a k-regime mean-switching variance model via MSwM.
#'
#' @param returns  numeric vector
#' @param k        number of regimes (default 3)
#' @return         list(model, smoothed_probs)
fit_mswm_baseline <- function(returns, k = 3) {
  df    <- data.frame(returns = returns)
  mod_lm <- lm(returns ~ 1, data = df)
  msm    <- tryCatch(
    msmFit(mod_lm, k = k, sw = c(TRUE, TRUE), data = df),
    error = function(e) { message("[R] MSwM failed: ", e$message); NULL }
  )
  if (is.null(msm)) return(NULL)
  list(
    model          = msm,
    transition_mat = msm@transMat,
    regime_params  = summary(msm)
  )
}


# =============================================================================
# 3. Bayesian Regime Model — rstanarm
# =============================================================================

#' Bayesian normal regression with regime indicator via rstanarm.
#' Uses a simple hierarchical structure as a baseline Bayesian model.
#'
#' @param returns  numeric vector
#' @param regimes  integer vector of regime labels (from HMM)
#' @param K        number of regimes
fit_rstanarm_regime <- function(returns, regimes, K = 5) {
  if (!has_rstanarm) {
    message("[R] rstanarm not available — skipping Bayesian regression")
    return(NULL)
  }
  
  df <- data.frame(
    ret    = returns,
    regime = factor(regimes, levels = 1:K)
  )
  
  fit <- tryCatch(
    rstanarm::stan_glm(
      ret ~ regime - 1,
      data   = df,
      family = gaussian(),
      prior  = rstanarm::normal(0, 0.02),
      prior_intercept = rstanarm::normal(0, 0.05),
      chains = 2,
      iter   = 1000,
      refresh = 0,
      seed   = 42
    ),
    error = function(e) { message("[R] rstanarm failed: ", e$message); NULL }
  )
  
  if (is.null(fit)) return(NULL)
  
  message("[R] rstanarm Bayesian regime model fitted")
  message("[R] R-hat diagnostics:")
  rhats <- summary(fit)[, "Rhat"]
  message(sprintf("  Max R-hat: %.4f | Min R-hat: %.4f",
                  max(rhats, na.rm=TRUE), min(rhats, na.rm=TRUE)))
  
  list(
    model   = fit,
    summary = summary(fit),
    rhat_ok = all(rhats < 1.05, na.rm = TRUE)
  )
}


# =============================================================================
# 4. Changepoint Detection — bcp and changepoint packages
# =============================================================================

#' Bayesian changepoint detection via bcp package.
#'
#' @param returns  numeric vector
#' @param burnin   MCMC burn-in
#' @param mcmc     MCMC draws
bcp_changepoints <- function(returns, burnin = 200, mcmc = 2000) {
  if (!has_bcp) {
    message("[R] bcp not available")
    return(NULL)
  }
  out <- bcp::bcp(returns, burnin = burnin, mcmc = mcmc)
  
  # Identify changepoints: posterior prob > 0.5
  cp_idx <- which(out$posterior.prob > 0.5)
  message(sprintf("[R] bcp detected %d changepoints (P > 0.5)", length(cp_idx)))
  
  list(
    model           = out,
    changepoints    = cp_idx,
    cp_probabilities = out$posterior.prob,
    posterior_means = out$posterior.mean
  )
}


#' PELT changepoint detection via changepoint package.
#'
#' @param returns  numeric vector
#' @param method   "PELT" or "AMOC"
#' @param penalty  "BIC" or "AIC"
pelt_changepoints <- function(returns, method = "PELT", penalty = "BIC") {
  if (!has_changepoint) {
    message("[R] changepoint not available")
    return(NULL)
  }
  
  cp_var  <- changepoint::cpt.var(returns, method = method, penalty = penalty)
  cp_mean <- changepoint::cpt.mean(returns, method = method, penalty = penalty)
  
  message(sprintf("[R] PELT variance changepoints: %d", length(cpts(cp_var))))
  message(sprintf("[R] PELT mean changepoints: %d", length(cpts(cp_mean))))
  
  list(
    var_changepoints  = changepoint::cpts(cp_var),
    mean_changepoints = changepoint::cpts(cp_mean),
    var_model         = cp_var,
    mean_model        = cp_mean
  )
}


# =============================================================================
# 5. Conformal Prediction in R (Section A6.2, A6.3, A6.5)
# =============================================================================

#' Split-conformal classifier: returns prediction sets with marginal coverage >= 1-alpha.
#'
#' @param cal_probs   matrix (n_cal, K) calibration probabilities
#' @param y_cal       integer vector (1-indexed) of true calibration labels
#' @param test_probs  matrix (n_test, K) test probabilities
#' @param alpha       miscoverage level (default 0.10)
#' @return list(sets, q_hat, coverage)
split_conformal_classifier_r <- function(cal_probs, y_cal, test_probs, alpha = 0.1) {
  n_cal <- nrow(cal_probs)
  K     <- ncol(cal_probs)
  
  # Non-conformity scores: 1 - p(true_class)
  cal_scores <- numeric(n_cal)
  for (i in 1:n_cal) {
    cal_scores[i] <- 1 - cal_probs[i, y_cal[i]]
  }
  
  # Conformal quantile (finite-sample correction)
  q_level <- min(ceiling((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
  q_hat   <- quantile(cal_scores, q_level, type = 1)
  
  # Prediction sets
  pred_sets <- test_probs >= (1 - q_hat)
  
  # Empirical calibration coverage
  coverage  <- mean(cal_scores <= q_hat)
  
  list(
    sets     = pred_sets,
    q_hat    = as.numeric(q_hat),
    coverage = coverage,
    set_sizes = rowSums(pred_sets)
  )
}


#' Adaptive Prediction Sets (Romano et al. 2020).
adaptive_prediction_sets_r <- function(cal_probs, y_cal, test_probs, alpha = 0.1) {
  n_cal <- nrow(cal_probs)
  K     <- ncol(cal_probs)
  
  # Sort probabilities high-to-low and compute cumulative sums
  cal_scores <- numeric(n_cal)
  for (i in 1:n_cal) {
    sorted_idx  <- order(cal_probs[i, ], decreasing = TRUE)
    sorted_probs <- cal_probs[i, sorted_idx]
    cumsums     <- cumsum(sorted_probs)
    # Score = cumsum up to and including the true class
    true_rank   <- which(sorted_idx == y_cal[i])
    cal_scores[i] <- cumsums[true_rank]
  }
  
  q_level <- min(ceiling((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
  q_hat   <- quantile(cal_scores, q_level, type = 1)
  
  # Test prediction sets
  pred_sets <- matrix(FALSE, nrow = nrow(test_probs), ncol = K)
  for (i in 1:nrow(test_probs)) {
    sorted_idx   <- order(test_probs[i, ], decreasing = TRUE)
    sorted_probs <- test_probs[i, sorted_idx]
    cumsums      <- cumsum(sorted_probs)
    in_set       <- cumsums <= q_hat
    in_set[1]    <- TRUE  # always include top class
    pred_sets[i, sorted_idx[in_set]] <- TRUE
  }
  
  list(
    sets      = pred_sets,
    q_hat     = as.numeric(q_hat),
    set_sizes = rowSums(pred_sets)
  )
}


#' Mondrian (class-conditional) conformal — separate threshold per class.
mondrian_conformal_r <- function(cal_probs, y_cal, test_probs, alpha = 0.1, K = 5) {
  q_hats <- numeric(K)
  for (k in 1:K) {
    mask_k   <- y_cal == k
    if (sum(mask_k) < 5) { q_hats[k] <- 1.0; next }
    scores_k <- 1 - cal_probs[mask_k, k]
    n_k      <- length(scores_k)
    q_level  <- min(ceiling((n_k + 1) * (1 - alpha)) / n_k, 1.0)
    q_hats[k] <- quantile(scores_k, q_level, type = 1)
  }
  
  pred_sets <- matrix(FALSE, nrow = nrow(test_probs), ncol = K)
  for (k in 1:K) {
    pred_sets[, k] <- test_probs[, k] >= (1 - q_hats[k])
  }
  
  list(sets = pred_sets, q_hats = q_hats, set_sizes = rowSums(pred_sets))
}


# =============================================================================
# 6. Python vs R Reconciliation
# =============================================================================

#' Compute numerical reconciliation metrics between Python and R outputs.
#'
#' @param python_probs  matrix (T, K) — Python HMM posterior probs
#' @param r_probs       matrix (T, K) — R depmixS4 posterior probs
#' @return data.frame with per-regime correlation and RMSE
reconcile_python_r <- function(python_probs, r_probs) {
  K <- ncol(python_probs)
  results <- lapply(1:K, function(k) {
    py <- python_probs[, k]
    r  <- r_probs[, k]
    corr <- cor(py, r)
    rmse <- sqrt(mean((py - r)^2))
    mae  <- mean(abs(py - r))
    data.frame(Regime = k, Correlation = round(corr, 4),
               RMSE = round(rmse, 6), MAE = round(mae, 6))
  })
  df <- do.call(rbind, results)
  message(sprintf("[R] Mean cross-language correlation: %.4f", mean(df$Correlation)))
  df
}


#' Expected Calibration Error in R.
ece_r <- function(probs_max, correct, n_bins = 10) {
  breaks <- seq(0, 1, length.out = n_bins + 1)
  n      <- length(probs_max)
  ece    <- 0
  for (i in 1:n_bins) {
    lo   <- breaks[i];  hi <- breaks[i + 1]
    mask <- probs_max > lo & probs_max <= hi
    if (sum(mask) == 0) next
    acc  <- mean(correct[mask])
    conf <- mean(probs_max[mask])
    ece  <- ece + (sum(mask) / n) * abs(acc - conf)
  }
  ece
}


# =============================================================================
# 7. Main Execution (run when script is sourced directly)
# =============================================================================

main_r <- function() {
  message("\n=== Bayesian Regime Detection Engine — R Codebase ===")
  message("CIN: U62012MH2023PTC410415")
  
  # Try to load data from synthetic CSV
  data_path <- file.path(dirname(sys.frame(1)$ofile), 
                          "..", "src", "data", "synthetic_indian_market.csv")
  
  if (file.exists(data_path)) {
    df      <- read.csv(data_path)
    returns <- diff(log(df$Close))
    message(sprintf("Loaded %d daily returns from %s", length(returns), data_path))
  } else {
    message("Synthetic data file not found — generating random returns for demo")
    set.seed(42)
    returns <- c(rnorm(200, 0.0008, 0.008),
                 rnorm(100, -0.0015, 0.035),
                 rnorm(150, -0.0005, 0.022),
                 rnorm(100, 0.0010, 0.012))
  }
  
  # 1. BIC comparison
  message("\n--- BIC Comparison (K=3,5,7) ---")
  bic_table <- compare_hmm_bic(returns, k_values = c(3, 5, 7))
  print(bic_table)
  
  # 2. Fit 5-state HMM
  message("\n--- Fitting 5-state Gaussian HMM (depmixS4) ---")
  hmm5 <- fit_hmm_depmixs4(returns, k = 5)
  if (!is.null(hmm5)) {
    message(sprintf("BIC: %.2f | AIC: %.2f", hmm5$bic, hmm5$aic))
    dur <- regime_durations(hmm5$states, K = 5)
    message("Regime Duration Statistics:")
    print(dur)
  }
  
  # 3. MSwM baseline
  message("\n--- MSwM Markov-Switching Baseline (K=3) ---")
  msm3 <- fit_mswm_baseline(returns, k = 3)
  if (!is.null(msm3)) {
    message("Transition matrix:")
    print(round(msm3$transition_mat, 4))
  }
  
  # 4. Changepoint detection
  message("\n--- PELT Changepoint Detection ---")
  pelt_res <- pelt_changepoints(returns)
  if (!is.null(pelt_res)) {
    message(sprintf("Variance changepoints: %s",
                    paste(pelt_res$var_changepoints, collapse=", ")))
    message(sprintf("Mean changepoints: %s",
                    paste(pelt_res$mean_changepoints, collapse=", ")))
  }
  
  # 5. Conformal demo
  message("\n--- Conformal Prediction Demo (mock probabilities) ---")
  set.seed(42)
  K       <- 5
  n_cal   <- 200
  n_test  <- 50
  cal_p   <- t(apply(matrix(rexp(n_cal * K), n_cal, K), 1, function(x) x / sum(x)))
  test_p  <- t(apply(matrix(rexp(n_test * K), n_test, K), 1, function(x) x / sum(x)))
  y_cal   <- sample(1:K, n_cal, replace = TRUE)
  
  sc_res  <- split_conformal_classifier_r(cal_p, y_cal, test_p, alpha = 0.10)
  message(sprintf("Split-Conformal: q_hat=%.4f, calibration_coverage=%.3f, avg_set_size=%.2f",
                  sc_res$q_hat, sc_res$coverage, mean(sc_res$set_sizes)))
  
  aps_res <- adaptive_prediction_sets_r(cal_p, y_cal, test_p, alpha = 0.10)
  message(sprintf("APS:             q_hat=%.4f, avg_set_size=%.2f",
                  aps_res$q_hat, mean(aps_res$set_sizes)))
  
  mond_res <- mondrian_conformal_r(cal_p, y_cal, test_p, alpha = 0.10)
  message(sprintf("Mondrian:        per-class q_hats: %s",
                  paste(round(mond_res$q_hats, 3), collapse=", ")))
  
  message("\n=== R Codebase Complete ===")
  message("CIN: U62012MH2023PTC410415")
}

# Run if executed as a script (not when sourced for testing)
if (!interactive()) {
  tryCatch(main_r(), error = function(e) {
    message("Error in main_r: ", e$message)
  })
}
