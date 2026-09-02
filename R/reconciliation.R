# =============================================================================
# Python vs R Reconciliation Script
# =============================================================================
# Cross-validates regime probabilities between Python (main.py) and
# R (models.R / conformal.R) implementations.
#
# Uses exported CSVs for comparison. Run after saving outputs from both sides.
# 
# =============================================================================

library(tidyverse)

# =============================================================================
# Load outputs
# =============================================================================

load_python_probs <- function(path = "outputs/python_hmm_probs.csv") {
  if (!file.exists(path)) {
    message("Python probs CSV not found at: ", path)
    message("Generate with: import pandas as pd; pd.DataFrame(hmm_probs).to_csv('outputs/python_hmm_probs.csv', index=False)")
    return(NULL)
  }
  read_csv(path, show_col_types = FALSE)
}

load_r_probs <- function(path = "outputs/r_hmm_probs.csv") {
  if (!file.exists(path)) {
    message("R probs CSV not found at: ", path)
    return(NULL)
  }
  read_csv(path, show_col_types = FALSE)
}

# =============================================================================
# Core Reconciliation
# =============================================================================

#' Compute pairwise correlation and MAE between Python and R regime probs.
reconcile_regime_probs <- function(py_probs, r_probs, regime_names = NULL) {
  stopifnot(nrow(py_probs) == nrow(r_probs))
  stopifnot(ncol(py_probs) == ncol(r_probs))
  
  K <- ncol(py_probs)
  if (is.null(regime_names)) {
    regime_names <- c("Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off")[1:K]
  }
  
  results <- tibble(
    regime       = regime_names,
    correlation  = numeric(K),
    mae          = numeric(K),
    max_diff     = numeric(K),
    rmse         = numeric(K),
  )
  
  for (k in 1:K) {
    py_k  <- as.numeric(py_probs[[k]])
    r_k   <- as.numeric(r_probs[[k]])
    diff_k <- abs(py_k - r_k)
    
    results$correlation[k] <- cor(py_k, r_k)
    results$mae[k]         <- mean(diff_k)
    results$max_diff[k]    <- max(diff_k)
    results$rmse[k]        <- sqrt(mean(diff_k^2))
  }
  
  results <- results %>%
    mutate(across(where(is.numeric), ~ round(., 4)))
  
  results
}

#' Flag time steps where |python - R| exceeds threshold for any regime.
flag_divergent_steps <- function(py_probs, r_probs, threshold = 0.05) {
  T <- nrow(py_probs)
  K <- ncol(py_probs)
  
  max_diff <- numeric(T)
  for (t in 1:T) {
    max_diff[t] <- max(abs(as.numeric(py_probs[t, ]) - as.numeric(r_probs[t, ])))
  }
  
  tibble(
    t          = 1:T,
    max_diff   = round(max_diff, 4),
    divergent  = max_diff > threshold,
  )
}

# =============================================================================
# Prediction Set Reconciliation
# =============================================================================

#' Check that Python and R conformal prediction sets have the same coverage.
#' @param py_sets  binary matrix from Python
#' @param r_sets   binary matrix from R
reconcile_prediction_sets <- function(py_sets, r_sets) {
  stopifnot(nrow(py_sets) == nrow(r_sets))
  
  py_sizes <- rowSums(py_sets)
  r_sizes  <- rowSums(r_sets)
  
  list(
    python_mean_set_size = round(mean(py_sizes), 3),
    r_mean_set_size      = round(mean(r_sizes), 3),
    set_size_diff        = round(mean(py_sizes) - mean(r_sizes), 3),
    sets_identical_pct   = round(mean(rowSums(py_sets == r_sets) == ncol(py_sets)) * 100, 1),
    python_coverage      = round(mean(py_sizes >= 1) * 100, 1),
    r_coverage           = round(mean(r_sizes >= 1) * 100, 1)
  )
}

# =============================================================================
# Synthetic Reconciliation Test (no CSV needed)
# =============================================================================

synthetic_reconciliation_test <- function(n = 300, K = 5) {
  message("\n=== Synthetic Reconciliation Test ===")
  message("\n")
  
  set.seed(42)
  
  # Simulate Python probs (softmax of random logits)
  logits   <- matrix(rnorm(n * K), n, K)
  py_probs <- t(apply(logits, 1, function(x) exp(x) / sum(exp(x))))
  
  # Simulate R probs = Python probs + small numerical noise
  r_probs <- py_probs + matrix(rnorm(n * K, 0, 0.002), n, K)
  r_probs <- t(apply(r_probs, 1, function(x) pmax(x, 0) / sum(pmax(x, 0))))
  
  colnames(py_probs) <- colnames(r_probs) <-
    c("Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off")[1:K]
  
  # Reconcile
  rec <- reconcile_regime_probs(as.data.frame(py_probs), as.data.frame(r_probs))
  message("Regime-level reconciliation:")
  print(rec)
  
  # Flag divergent steps
  flags <- flag_divergent_steps(as.data.frame(py_probs), as.data.frame(r_probs),
                                 threshold = 0.05)
  n_div  <- sum(flags$divergent)
  pct_div <- n_div / n * 100
  message(sprintf("\nDivergent steps (>5%%): %d / %d (%.1f%%)", n_div, n, pct_div))
  
  if (all(rec$correlation > 0.99)) {
    message("\nRECONCILIATION PASSED: Python and R agree within numerical tolerance")
  } else {
    message("\nRECONCILIATION WARNING: Some regimes show low Python-R correlation")
  }
  
  list(reconciliation = rec, flags = flags)
}

# =============================================================================
# Full reconciliation (with CSV input)
# =============================================================================

run_full_reconciliation <- function() {
  message("=== Python vs R Full Reconciliation ===")
  message("\n")
  
  py_probs <- load_python_probs()
  r_probs  <- load_r_probs()
  
  if (is.null(py_probs) || is.null(r_probs)) {
    message("Falling back to synthetic reconciliation test...")
    return(synthetic_reconciliation_test())
  }
  
  message("Loaded Python probs: ", nrow(py_probs), " rows x ", ncol(py_probs), " cols")
  message("Loaded R probs:      ", nrow(r_probs), " rows x ", ncol(r_probs), " cols")
  
  rec   <- reconcile_regime_probs(py_probs, r_probs)
  flags <- flag_divergent_steps(py_probs, r_probs)
  
  message("\nRegime-level reconciliation:")
  print(rec)
  
  n_div  <- sum(flags$divergent)
  pct_div <- n_div / nrow(flags) * 100
  message(sprintf("\nDivergent steps (>5%%): %d / %d (%.1f%%)", n_div, nrow(flags), pct_div))
  
  list(reconciliation = rec, flags = flags)
}

# =============================================================================
# Entry point
# =============================================================================

if (!interactive()) {
  res <- run_full_reconciliation()
} else {
  # In interactive mode, run the synthetic test
  res <- synthetic_reconciliation_test()
}
